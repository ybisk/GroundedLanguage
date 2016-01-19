import json
import math
import gzip
import sys
import random

random.seed(11032015)


############################ Vector Operations ################################
def vec(x, y):
  return [round(y[0] - x[0], 3), round(y[1] - x[1], 3), round(y[2] - x[2], 3)]


def angle(v):
  # If the vector has no magnitude
  if size(v) == 0:
    return 0
  # compute from [1,1,1] == acos(V.1 / |V||1|)
  # acos ( sum(V) / |V|*sqrt(3) )
  return round(math.acos(sum(v) / (size(v) * math.sqrt(3))), 2)


def strvec(x):
  return "%5.2f %5.2f %5.2f" % (x[0], x[1], x[2])


def size(vector):
  s = 0.0
  for v in vector:
    s += v ** 2
  return round(math.sqrt(s), 2)


def shrink(pos):
  return [0 if round(float(v), 4) == 0 else round(float(v), 4) for v in pos.split(",")[:3]]


############################ String Operations ################################

def clean_note(toclean):
  clean = toclean.lower()
  clean = clean.replace(", ", " , ")
  clean = clean.replace(".", " . ")
  clean = clean.replace(": ", " : ")
  clean += " "
  # for num in range(21):
  #  note = note.replace(" %d " % num, " _%d_ " % num)
  return clean


############################ State Encodings ##################################


# Concatenate the locations of all the blocks (in order)
def world(arr):
  worldvec = []
  for block in arr:
    worldvec.extend(block[1])
  return worldvec


# Action representation
def semantics(time_t, time_tp1):
  # Find the largest change
  maxval = 0
  for b_id in range(len(time_t)):
    # If the state has changed
    if time_t[b_id] != time_tp1[b_id]:
      before = time_t[b_id]
      after = time_tp1[b_id]

      pos_dist = size(vec(before[1], after[1]))
      if pos_dist > maxval:
        changed = after
        maxval = pos_dist

  return {"id": changed[0], "loc": changed[1]}


############################ Generating New Examples ##########################

def shift(state, x, z):
  newstate = []
  for bid, pos in state:
    newstate.append((bid, [pos[0] + x, pos[1], pos[2] + z]))
  return newstate


def displace(tuples):
  newtuples = []
  newtuples.extend(tuples)
  factor = 1
  while factor < 100:
    print "Displacing: ", factor
    factor += 1
    x = random.random() - 0.5
    z = random.random() - 0.5
    for t in tuples:
      before = shift(t[0], x, z)
      after = shift(t[1], x, z)
      notes = t[2]
      newtuples.append((before, after, notes))
  print len(tuples), " -> ", len(newtuples)
  return newtuples


############################ Main Body ########################################


def shrinkWorld(W):
  current = {}
  for ID in range(len(W)):
    block = W[ID]
    current[block["id"]] = shrink(block["position"])
  return [(ID,current[ID]) for ID in range(1,len(W)+1)]

anno_worlds = gzip.open(sys.argv[1],'r')
inittuples = []
for line in anno_worlds:
  j = json.loads(line)
  before = shrinkWorld(j["current"])
  after  = shrinkWorld(j["next"])
  utterance = clean_note(j["utterance"])
  inittuples.append((before, after, utterance))


print "Read Data"

# Shift everything around on the board slightly to generate artificial data
newTuples = []
#addDisplacement = True
#if addDisplacement:
#  newTuples = displace(inittuples)

# Training Files
#source = gzip.open("out/source.json.gz", 'w')
#target = gzip.open("out/target.json.gz", 'w')
#num = 0
#for before, after, note in newTuples:
#  source.write(json.dumps({"ex": num, "world": world(before), "text": note}) + "\n")
#  sem = semantics(before, after);
#  sem["ex"] = num
#  target.write(json.dumps(sem) + "\n")
#  num += 1
#source.close()
#target.close()

# Training Files
source = gzip.open("out/source.orig.json.gz", 'w')
target = gzip.open("out/target.orig.json.gz", 'w')
num = 0
for before, after, note in inittuples:
  source.write(json.dumps({"ex": num, "world": world(before), "text": note}) + "\n")
  sem = semantics(before, after)
  sem["ex"] = num
  target.write(json.dumps(sem) + "\n")
  num += 1
source.close()
target.close()
