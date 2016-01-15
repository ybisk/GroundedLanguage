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
ploc= []
for line in open(sys.argv[1],'r'):
  line = line.strip()
  if len(line) != 0 and line[0] == "A":
    if len(pS) == 0:
      pS = [int(v) for v in line[4:len(line)-1].split(",")]
    elif len(ploc) == 0:
      ploc = [float(v) for v in line[4:len(line)-1].split(",")]

c = 0
for i in range(len(pS)):
  if pS[i] == Sources[i]:
    c += 1
print "Grounding: ", 100.0*c/len(pS)

err = []
off = 0.1666
for i in range(len(pS)):
  loc = [ploc[3*i], ploc[3*i + 1], ploc[3*i + 2]]
  err.append(dist(loc, Goals[i]))

err.sort()
print "Mean", sum(err)/len(err)
print "Median", err[len(err)/2]
