import sys,json,gzip,copy
from nltk.tokenize import TreebankWordTokenizer

directory = sys.argv[1]

def Act(world, idx, loc):
  idx -= 1
  newWorld = copy.deepcopy(world)
  newWorld[3*idx] = loc[0]
  newWorld[3*idx + 1] = loc[1]
  newWorld[3*idx + 2] = loc[2]
  return newWorld


## Convert Each Section to Matrix ##
for section in ["Train","Dev","Test"]:

  Text = []
  World = []
  Type = []
  for line in gzip.open("%s/%s.input.orig.json.gz" % (directory,section),'r'):
    j = json.loads(line)
    text = TreebankWordTokenizer().tokenize(j["text"].lower())
    Text.append(text)
    World.append(j["world"])
    if "decoration" in j:
      Type.append(j["decoration"])

  source = []
  locs = []
  NewWorld = []
  c = 0
  for line in gzip.open("%s/%s.output.orig.json.gz" % (directory,section),'r'):
    j = json.loads(line)
    source.append(j["id"])
    locs.append(j["loc"])
    goal_location = j["loc"]
    NewWorld.append(Act(World[c], j["id"], j["loc"]))
    c += 1

  Sout = open("%s/%s.WW-T.source" % (directory, section),'w')
  Tout = open("%s/%s.WW-T.target" % (directory, section),'w')
  for i in range(len(source)):
    Sout.write("%s %s %s\n" % (Type[i], ' '.join("%-4.2f" % v for v in World[i]), ' '.join("%-4.2f" % v for v in NewWorld[i])))
    Tout.write("%s\n" % Text[i])
  Sout.close()
  Tout.close()
