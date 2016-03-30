import sys,gzip,json,math

def dist(x, y):
  return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2) / 0.1524

Worlds = []
for line in gzip.open("BlockWorld/%s/Dev.input.orig.json.gz" % sys.argv[1],'r'):
  Worlds.append(json.loads(line)["world"])

Sources = []
Goals = []
for line in gzip.open("BlockWorld/%s/Dev.output.orig.json.gz" % sys.argv[1],'r'):
  j = json.loads(line)
  Sources.append(j["id"])
  Goals.append(j["loc"])


Dev = open("BlockWorld/%s/Dev.STRP.data" % sys.argv[1],'r')
pS = []
pT = []
pRP = []
for line in Dev:
  line = line.split()
  pS.append(int(line[len(line)-3]))
  pT.append(int(line[len(line)-2]))
  pRP.append(int(line[len(line)-1]))

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
    print "Error, Invalid\n"
    sys.exit()
  err.append(dist(loc, Goals[i]))

err.sort()
print "Mean", sum(err)/len(err)
print "Median", err[len(err)/2]
