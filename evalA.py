import os,sys,json,gzip,math

def dist(x, y):
  return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2) / 0.1524

Worlds = []
for line in gzip.open("BlockWorld/logos/Dev.input.orig.json.gz",'r'):
  Worlds.append(json.loads(line)["world"])

Sources = []
Goals = []
for line in gzip.open("BlockWorld/logos/Dev.output.orig.json.gz",'r'):
  j = json.loads(line)
  Sources.append(j["id"])
  Goals.append(j["loc"])

pS = []
pT = []
pRP= []
for line in open(sys.argv[1],'r'):
  if line[0] == "[":
    line = line.strip()
    line = [int(v) for v in line[1:len(line)-1].split(",")]
    if len(pS) == 0:
      pS = line
    elif len(pT) == 0:
      pT = line
    elif len(pRP) == 0:
      pRP = line

c = 0
for i in range(len(pS)):
  if pS[i] == Sources[i]:
    c += 1
print "Grounding: ", 100.0*c/len(pS)

err = []
off = 0.1666
for i in range(len(pT)):
  target = pT[i] - 1
  if target*3 >= len(Worlds[i]):
    loc = [0.0,0.1,0.0]
  else:
    #print target, len(Worlds[i]), Worlds[i]
    loc = [Worlds[i][target*3], Worlds[i][target*3 + 1], Worlds[i][target*3 + 2]]
  RP = pRP[i]
  if RP == 1:
    loc[0] += off
    loc[2] += off
  elif RP == 2:
    loc[0] += off
  elif RP == 3:
    loc[0] += off
    loc[2] -= off
  elif RP == 4:
    loc[2] -= off
  elif RP == 5:
    loc[0] -= off
    loc[2] -= off
  elif RP == 6:
    loc[0] -= off
  elif RP == 7:
    loc[0] -= off
    loc[2] += off
  elif RP == 8:
    loc[2] += off
  else:
    print "Error, Invalid\n",words,brands[act],brands[targetblock],loc,goal_location
    sys.exit()
  
  err.append(dist(loc, Goals[i]))

err.sort()
print "Mean", sum(err)/len(err)
print "Median", err[len(err)/2]
