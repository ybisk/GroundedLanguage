import sys,json,gzip,copy,math
import editdistance
from nltk.tokenize import TreebankWordTokenizer

directory = sys.argv[1]

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

# Cardinal directions
cardinal = ["southwest", "west", "northwest", "north", "northeast", "east", "southeast", "south"]
relative = ["down to the left", "left", "up and left", "above", "above to the right", "right", "down and right", "below"]
relativePrime = ["left and down", "left", "left and above", "above", "diagonally right and above", "right", "right and below", "below"]

## Convert Each Section to Matrix ##
for section in ["Train","Dev","Test"]:

  Text = []
  World = []
  Type = []
  target = []
  RP = []
  f = "%s/%s.input.orig.json.gz" % (directory,section)
  for line in gzip.open(f,'r'):
    j = json.loads(line)
    text = TreebankWordTokenizer().tokenize(j["text"].lower())
    Text.append(text)
    World.append(extend(j["world"]))
    if "decoration" in j:
      Type.append(j["decoration"])

  source = []
  locs = []
  NewWorld = []
  c = 0
  f = "%s/%s.output.orig.json.gz" % (directory,section)
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

    usinglogos = True if ((len(Type) > 0 and Type[c] == "logo") or directory == "logos") else False
    #print usinglogos, Type[c] if len(Type) > 0 else ""
    usinglogos = True
    if usinglogos:
      blocks = logoblocks
    else:
      blocks = digitblocks
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
      print "Error, Invalid\n",words,brands[act],brands[targetblock],loc,goal_location
      sys.exit()

    NewWorld.append(Act(World[c], j["id"], j["loc"]))
    c += 1

  out = gzip.open("%s/%s.template_based_generation.data.gz" % (directory, section),'w')
  orig = gzip.open("%s/%s.original_text.data.gz" % (directory, section), 'w')
  for i in range(len(source)):
    out.write("move %s to the %s of %s\n" % (brands[source[i] - 1] if usinglogos else digits[source[i] - 1],
                                             cardinal[RP[i] - 1],
                                             brands[target[i] - 1] if usinglogos else digits[target[i] - 1]))
    orig.write(" ".join(Text[i]) + "\n")
    out.write("place %s %s of %s\n" % (brands[source[i] - 1] if usinglogos else digits[source[i] - 1],
                                             relative[RP[i] - 1],
                                             brands[target[i] - 1] if usinglogos else digits[target[i] - 1]))
    orig.write(" ".join(Text[i]) + "\n")
    out.write("place %s %s of %s\n" % (brands[source[i] - 1] if usinglogos else digits[source[i] - 1],
                                       relativePrime[RP[i] - 1],
                                       brands[target[i] - 1] if usinglogos else digits[target[i] - 1]))
    orig.write(" ".join(Text[i]) + "\n")
  orig.close()
  out.close()
