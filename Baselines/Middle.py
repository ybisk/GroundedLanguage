import sys,gzip,json,math

def dist(x, y):
  return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2) / 0.1524

Worlds = []
for line in gzip.open("BlockWorld/digits/Dev.input.orig.json.gz",'r'):
  Worlds.append(json.loads(line)["world"])

Sources = []
Goals = []
for line in gzip.open("BlockWorld/digits/Dev.output.orig.json.gz",'r'):
  j = json.loads(line)
  Sources.append(j["id"])
  Goals.append(j["loc"])


Dev = open("BlockWorld/digits/Dev.STRP.data",'r')
pS = []
for line in Dev:
  line = line.split()
  pS.append(int(line[len(line)-3]))

predicted_locs = [[0,0.1,0]] * len(pS)

c = 0
for i in range(len(pS)):
  if pS[i] == Sources[i]:
    c += 1
print "Grounding: ", 100.0*c/len(pS)

err = []
off = 0.1666
for i in range(len(predicted_locs)):
  err.append(dist(predicted_locs[i], Goals[i]))

err.sort()
print "Mean", sum(err)/len(err)
print "Median", err[len(err)/2]
