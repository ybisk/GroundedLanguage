import os,sys,json,gzip,codecs
import editdistance
from nltk.tokenize import TreebankWordTokenizer

directory = sys.argv[1]

def extend(world):
  while len(world) < 60:
    world.append(-1)
  return world

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
    World.append(extend(j["world"]))

  source = []
  RP = []
  locs = []
  c = 0
  for line in gzip.open("%s/%s.output.orig.json.gz" % (directory,section),'r'):
    j = json.loads(line)
    source.append(j["id"])
    locs.append(j["loc"])
    c += 1

  out = open("%s/%s.SP.data" % (directory, section),'w')
  for i in range(len(source)):
    out.write("%s %s %d %s\n" % (' '.join("%-3s" % v for v in Hot[i]), ' '.join("%-5.3f" % v for v in World[i]), source[i], ' '.join("%5.3f" % v for v in locs[i])))
  out.close()
