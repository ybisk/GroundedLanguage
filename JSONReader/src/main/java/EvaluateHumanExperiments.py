import os,sys,json,ast,math

if len(sys.argv) < 3:
  print "python EvaluateHumanExperiments.py HumanDir GoldDir"
  sys.exit()

def findSource(X,Y):
  for i in range(len(X)):
    if X[i] != Y[i]:
      return i
  return -1

def correctSource(source, gold, pred):
  g = gold[source]
  p = ast.literal_eval(pred[source]["position"])
  d = math.sqrt((g[0] - p[0])**2 + (g[1] - p[1])**2 + (g[2] - p[2])**2)/0.1524
  if d > 0.05:
    return True
  return False

def diff(source, gold, pred):
  g = gold[source]
  p = ast.literal_eval(pred[source]["position"])
  return math.sqrt((g[0] - p[0])**2 + (g[1] - p[1])**2 + (g[2] - p[2])**2)/0.1524

Human = {}
Task = {}
for root, folders, files in os.walk(sys.argv[1]):
  for name in files:
    j = json.load(open(os.path.join(root, name), 'r'))
    Human[j["name"].replace("_world","")] = j["block_state"]
    Task[j["name"].replace("_world","")] = j["start_id"]

Gold = {}
GoldBefore = {}
for root, folders, files in os.walk(sys.argv[2]):
  for name in files:
    j = json.load(open(os.path.join(root,name), 'r'))
    Gold[j["name"].replace("_gold.json","")] = j["next"]
    GoldBefore[j["name"].replace("_gold.json","")] = j["current"]

print len(Human), len(Gold)

Errors = []
source = 0
for name in Gold:
  g = Gold[name]
  p = Human[name]
  changed = findSource(Gold[name],GoldBefore[name])
  if correctSource(changed, g, p):
    source += 1
  else:
    print "Failed ", name, Task[name]
  Errors.append(diff(changed, g, p))

print "Mean: ", sum(Errors)/len(Errors)
Errors.sort()
print "Median: ", Errors[len(Errors)/2]
print "Source: ", source, "/25"