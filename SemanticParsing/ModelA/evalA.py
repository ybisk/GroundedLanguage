import sys,json,gzip,math

def dist(x, y):
  return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2) / 0.1524

folder = "logos" if len(sys.argv) == 2 else sys.argv[2]
verbose = False

# Set of brands for labeling blocks
brands = [
    'adidas', 'bmw', 'burger king', 'coca cola', 'esso',
    'heineken', 'hp', 'mcdonalds', 'mercedes benz', 'nvidia',
    'pepsi', 'shell', 'sri', 'starbucks', 'stella artois',
    'target', 'texaco', 'toyota', 'twitter', 'ups']

# Set of digits for labeling blocks
digits = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
    'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen'
    'nineteen', 'twenty'
]

dirs = ['NE','E','SE','S','SW','W','NW','N']

Worlds = []
Sents = []
for line in gzip.open("BlockWorld/%s/Dev.input.orig.json.gz" % folder,'r'):
  j = json.loads(line)
  Worlds.append(j["world"])
  Sents.append(j["text"])

Sources = []
Goals = []
for line in gzip.open("BlockWorld/%s/Dev.output.orig.json.gz" % folder,'r'):
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


if verbose:
  S = []
  T = []
  RP = []
  for line in open("BlockWorld/%s/Dev.STRP.data" % folder,'r'):
    line = line.strip().split()
    S.append(int(line[len(line)-3]))
    T.append(int(line[len(line)-2]))
    RP.append(int(line[len(line)-1]))
  o = open("errors.txt",'w')
  for i in range(len(pT)):
    o.write("%-1d %-1d %-1d %-15s %-15s %-2s  %-s\n" %
      (1 if pS[i] == S[i] else 0,
       1 if pT[i] == T[i] else 0,
       1 if pRP[i] == RP[i] else 0,
      brands[pS[i]-1] if folder == "logos" else digits[pS[i]-1],
      brands[pT[i]-1] if folder == "logos" else digits[pT[i]-1],
      dirs[pRP[i]-1], Sents[i]))
  o.close()
