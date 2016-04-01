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

def predicted(source, gold, pred):
  for i in range(len(gold)/3):
    if diff(i,gold,pred) > 0.05 and i != source:
      return i
  return -1

def SourceDiff(before, source, predicted):
  g = before[source]
  p = before[predicted]
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
SErrors = []
source = 0
for name in Gold:
  g = Gold[name]
  p = Human[name]
  goldsource = findSource(Gold[name],GoldBefore[name])
  if correctSource(goldsource, g, p):
    source += 1
    SErrors.append(0.0)
  else:
    print "Failed ", name, Task[name]
    predictedSource = predicted(goldsource, Gold[name], p)
    SErrors.append(SourceDiff(GoldBefore[name], goldsource, predictedSource))
  Errors.append(diff(goldsource, g, p))

print "Target"
print "Mean: ", sum(Errors)/len(Errors)
Errors.sort()
print "Median: ", Errors[len(Errors)/2]
print Errors
print "\nSource: ", source, "/25"
print "Mean: ", sum(SErrors)/len(SErrors)
Errors.sort()
print "Median: ", SErrors[len(SErrors)/2]
print SErrors
