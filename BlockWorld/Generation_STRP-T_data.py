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
  #if section == "Train":
  #  trainfile = "%s/%s.input.auto.json.gz" % (directory,section)
  #else:
  trainfile = "%s/%s.input.orig.json.gz" % (directory,section)
  for line in gzip.open(trainfile,'r'):
    j = json.loads(line)
    text = TreebankWordTokenizer().tokenize(j["text"].lower())
    Text.append(text)
    World.append(j["world"])
    Type.append("logo")
    if "decoration" in j:
      Type.append(j["decoration"])

  source = []
  locs = []
  NewWorld = []
  c = 0
  #if section == "Train":
  #  trainout = "%s/%s.output.auto.json.gz" % (directory,section)
  #else:
  trainout = "%s/%s.output.orig.json.gz" % (directory,section)
  for line in gzip.open(trainout,'r'):
    j = json.loads(line)
    source.append(j["id"])
    locs.append(j["loc"])
    goal_location = j["loc"]
    NewWorld.append(Act(World[c], j["id"], j["loc"]))
    c += 1

  Sout = open("%s/%s.WA-T1.source" % (directory, section),'w')
  Tout = open("%s/%s.WA-T1.target" % (directory, section),'w')
  print len(source), len(Type), len(World), len(locs)
  for i in range(len(source)):
    Sout.write("%s %s %d %s\n" %
               (Type[i], ' '.join("%-4.1f" % v for v in World[i]), source[i],
                ' '.join("%-4.1f" % v for v in locs[i])))
    Tout.write("%s\n" % ' '.join(Text[i]))
  Sout.close()
  Tout.close()
