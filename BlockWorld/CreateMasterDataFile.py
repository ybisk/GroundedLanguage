import sys,json,gzip,copy,math,codecs
import editdistance
from nltk.tokenize import TreebankWordTokenizer

directory = sys.argv[1]

longest = 0
### Read Vocabulary ###
Vocab = {"<unk>":1}
for line in codecs.open(directory + "/Vocab.txt",'r','utf-8'):
  line = line.split()
  if int(line[1]) >= 5:
    Vocab[line[0]] = len(Vocab) + 1

def integer(utr):
  v = []
  for i in range(longest):
    if i >= len(utr) or utr[i] not in Vocab:
      v.append(1)
    else:
      v.append(Vocab[utr[i]])
  return [str(i) for i in v]

def Act(world, idx, loc):
  idx -= 1
  newWorld = copy.deepcopy(world)
  newWorld[3*idx] = loc[0]
  newWorld[3*idx + 1] = loc[1]
  newWorld[3*idx + 2] = loc[2]
  return newWorld

def extend(world):
  while len(world) < 60:
    world.append(-1)
  return world

def distance((x, y, z), (a, b, c)):
  return math.sqrt((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2) / 0.1524

def Grid(loc):
  x = int(loc[0]/0.1528)
  y = int(loc[1]/0.1528)
  z = int(loc[2]/0.1528)
  return x + 13*y + 169+z

# Set of brands for labeling blocks
brands = [
    'adidas', 'bmw', 'burger king', 'coca cola', 'esso',
    'heineken', 'hp', 'mcdonalds', 'mercedes benz', 'nvidia',
    'pepsi', 'shell', 'sri', 'starbucks', 'stella artois',
    'target', 'texaco', 'toyota', 'twitter', 'ups']

# Set of digits for labeling blocks
digits = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen'
    'nineteen', 'twenty'
]


f = "%s/Train.input.decor.json.gz" % (directory)
for line in gzip.open(f,'r'):
  j = json.loads(line)
  text = TreebankWordTokenizer().tokenize(j["text"].lower())
  if len(text) > longest:
    longest = len(text)


## Convert Each Section to Matrix ##
for section in ["Train","Dev","Test"]:

  Text = []
  World = []
  Type = []
  target = []
  RP = []
  Hot = []
  f = "%s/%s.input.decor.json.gz" % (directory,section)
  for line in gzip.open(f,'r'):
    j = json.loads(line)
    text = TreebankWordTokenizer().tokenize(j["text"].lower())
    Text.append(text)
    Hot.append(integer(text))
    World.append(extend(j["world"]))
    if "decoration" in j:
      Type.append(j["decoration"])

  source = []
  locs = []
  NewWorld = []
  grid = []
  c = 0
  f = "%s/%s.output.decor.json.gz" % (directory,section)
  for line in gzip.open(f,'r'):
    j = json.loads(line)
    source.append(j["id"])
    locs.append(j["loc"])
    goal_location = j["loc"]
    words = Text[c]

    logoblocks = set()
    digitblocks = set()
    ## First, try and find brands ##
    for brand in brands:
      brandparts = brand.split()
      for part in brandparts:
        for word in words:
          if part == word or (len(word) > 2 and word != "up1" and editdistance.eval(part, word) < 2):
            logoblocks.add(brands.index(brand) + 1)

    ## Second, try and find numbers
    for digit in digits:
      for word in words:
        if editdistance.eval(digit, word) < 2:
          digitblocks.add(digits.index(digit) + 1)
    for digit in range(1,21):
      if str(digit) in words:
        digitblocks.add(digit)

    act = j["id"]

    decoration = j["decoration"]
    if decoration == "logos":
      blocks = logoblocks
    elif decoration == "digits":
      blocks = digitblocks
    else:
      blocks = set()
      for digit in range(1,len(World[c])/3 + 1):
        blocks.add(digit)
    if act in blocks:
      blocks.remove(act)
    ## Possible reference blocks
    if len(blocks) > 0:
      d = 100000
      for block in blocks:
        loc = World[c][3 * (block - 1)], World[c][3 * (block - 1) + 1], World[c][3 * (block - 1) + 2]
        dist = distance(loc, goal_location)
        if dist < d:
          d = dist
          targetblock = block
    else:
      targetblock = act
    target.append(targetblock)
    loc = World[c][3 * (targetblock - 1)], World[c][3 * (targetblock - 1) + 1], World[c][3 * (targetblock - 1) + 2]

    # Discretize
    if loc[0] <    goal_location[0] and loc[2] <  goal_location[2]:     # SW
      RP.append(1)
    elif loc[0] <  goal_location[0] and loc[2] == goal_location[2]:  # W
      RP.append(2)
    elif loc[0] <  goal_location[0] and loc[2] >  goal_location[2]:   # NW
      RP.append(3)
    elif loc[0] == goal_location[0] and loc[2] >  goal_location[2]:  # N
      RP.append(4)
    elif loc[0] >  goal_location[0] and loc[2] >  goal_location[2]:   # NE
      RP.append(5)
    elif loc[0] >  goal_location[0] and loc[2] == goal_location[2]:  # E
      RP.append(6)
    elif loc[0] >  goal_location[0] and loc[2] <  goal_location[2]:   # SE
      RP.append(7)
    elif loc[0] == goal_location[0] and loc[2] <  goal_location[2]:  # S
      RP.append(8)
    else:
      if decoration == "blank":
        RP.append(9)
      else:
        print "Error, Invalid\n",words,brands[act],brands[targetblock],loc,goal_location
        sys.exit()

    grid.append(Grid(goal_location))

    NewWorld.append(Act(World[c], j["id"], j["loc"]))
    c += 1

  out = gzip.open("%s/%s.WWT-STRPLocGrid.data.gz" % (directory, section),'w')
  for i in range(len(source)):
    out.write("%s %s %s %d %d %d %s %d\n" % (
                                    ' '.join("%-4.2f" % v for v in World[i]),    # Wt
                                    ' '.join("%-4.2f" % v for v in NewWorld[i]), # Wt+1
                                    ' '.join("%-3s" % v for v in Hot[i]),     # Text
                                    source[i], target[i], RP[i],              # STRP
                                    ' '.join("%5.3f" % v for v in locs[i]),   # Loc
                                    grid[i]
              ))
  out.close()

print "nx", longest
print "xvocab", len(Vocab)
