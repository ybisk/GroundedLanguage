import math,sys,json,gzip,codecs
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


def distance((x, y, z), (a, b, c)):
  return math.sqrt((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2) / 0.1524

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
  Type = []
  for line in gzip.open("%s/%s.input.orig.json.gz" % (directory,section),'r'):
    j = json.loads(line)
    text = TreebankWordTokenizer().tokenize(j["text"].lower())
    Hot.append(integer(text))
    Text.append(text)
    World.append(j["world"])
    if "decoration" in j:
      Type.append(j["decoration"])

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

    blocks = set()
    ## Second, try and find numbers
    for digit in range(1,len(World[c])/3 + 1):
        blocks.add(digit)

    act = j["id"]
    blocks.remove(act)
    ## Possible reference blocks
    d = 100000
    for block in blocks:
      loc = World[c][3 * (block - 1)], World[c][3 * (block - 1) + 1], World[c][3 * (block - 1) + 2]
      dist = distance(loc, goal_location)
      if dist < d:
        d = dist
        targetblock = block
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
      RP.append(9)
    c += 1

  out = open("%s/%s.STRP.data" % (directory, section),'w')
  for i in range(len(source)):
    out.write("%s %d %d %d\n" % (' '.join("%-3s" % v for v in Hot[i]),source[i],target[i],RP[i]))
  out.close()
