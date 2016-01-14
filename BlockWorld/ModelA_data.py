import os,sys,json,gzip,codecs
import editdistance
from nltk.tokenize import TreebankWordTokenizer

directory = sys.argv[1]

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

## Compute Longest Sentence ##
for line in gzip.open("%s/%s.input.orig.json.gz" % (directory,"Train"),'r'):
  j = json.loads(line)
  text = TreebankWordTokenizer().tokenize(j["text"])
  if len(text) > longest:
    longest = len(text)

## Convert Each Section to Matrix ##
for section in ["Train","Dev","Test"]:

  Hot = []
  Text = []
  World = []
  for line in gzip.open("%s/%s.input.orig.json.gz" % (directory,section),'r'):
    j = json.loads(line)
    text = TreebankWordTokenizer().tokenize(j["text"].lower())
    Hot.append(integer(text))
    Text.append(text)
    World.append(j["world"])

  source = []
  target = []
  RP = []
  locs = []
  c = 0
  for line in gzip.open("%s/%s.output.orig.json.gz" % (directory,section),'r'):
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

    usinglogos = True if len(logoblocks) >= len(digitblocks) else False
    if usinglogos:
      blocks = logoblocks
    else:
      blocks = digitblocks
    if act in blocks:
      blocks.remove(act)
    ## Possible reference blocks
    if len(blocks) > 0:
      targetblock = blocks.pop()
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
    c += 1

  out = open("%s/%s.STRP.data" % (directory, section),'w')
  for i in range(len(source)):
    out.write("%s %d %d %d\n" % (' '.join("%-3s" % v for v in Hot[i]),source[i],target[i],RP[i]))
  out.close()
