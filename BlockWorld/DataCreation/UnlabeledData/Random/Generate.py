import os, sys, json, math, copy
import random

faces = {"shape_params": {"face_4": {"color": "magenta", "orientation": "1"},
                          "face_5": {"color": "yellow", "orientation": "1"},
                          "face_6": {"color": "red", "orientation": "2"},
                          "face_1": {"color": "blue", "orientation": "1"},
                          "face_2": {"color": "green", "orientation": "1"},
                          "face_3": {"color": "cyan", "orientation": "1"}, "side_length": "0.1524"}, "type": "cube",
         "size": 0.5}
shape = {"block_meta": {"decoration": "blank", "blocks": [
  {"shape": faces, "name": "adidas", "id": 1},
  {"shape": faces, "name": "bmw", "id": 2},
  {"shape": faces, "name": "burger king", "id": 3},
  {"shape": faces, "name": "coca cola", "id": 4},
  {"shape": faces, "name": "esso", "id": 5},
  {"shape": faces, "name": "heineken", "id": 6},
  {"shape": faces, "name": "hp", "id": 7},
  {"shape": faces, "name": "mcdonalds", "id": 8},
  {"shape": faces, "name": "mercedes benz", "id": 9},
  {"shape": faces, "name": "nvidia", "id": 10}]}}


# {"shape": faces, "name": "pepsi", "id": 11},
# {"shape": faces, "name": "shell", "id": 12},
# {"shape": faces, "name": "sri", "id": 13},
# {"shape": faces, "name": "starbucks", "id": 14},
# {"shape": faces, "name": "stella artois", "id": 15},
# {"shape": faces, "name": "target", "id": 16},
# {"shape": faces, "name": "texaco", "id": 17},
# {"shape": faces, "name": "toyota", "id": 18},
# {"shape": faces, "name": "twitter", "id": 19}]}}

def dist(ax, ay, az, x, y, z):
  return math.sqrt((ax - x) ** 2 + (ay - y) ** 2 + (az - z) ** 2)

def intersection(bl, locs):
  for bid, bloc in locs:
    if bloc[0] - 0.1524 < bl[0] < bloc[0] + 0.1524 and bloc[2] - 0.1524 < bl[2] < bloc[2] + 0.1524:
      return True
  return False

def randomlocation(locations, block=None):
  x = random.random() * 1.6 - 0.8
  y = 0.0762
  #y = 0.1
  z = random.random() * 1.6 - 0.8
  for bid, (bx, by, bz) in locations:
    if dist(bx, by, bz, x, y, z) < 0.2:
      return randomlocation(locations)
  return [x, y, z]


def towerlocation(locations, block=None):
  rid, buildon = random.sample(locations, 1)[0]
  while rid == block:
    rid, buildon = random.sample(locations, 1)[0]
  for bid, (bx, by, bz) in locations:
    if bx == buildon[0] and bz == buildon[2] and by > buildon[1]:
      buildon = [bx, by, bz]
  return [buildon[0], buildon[1] + 0.1524, buildon[2]]

def walllocation(locations, block=None):
  rid, buildon = random.sample(locations, 1)[0]
  while rid == block or buildon[1] > 0.1:
    rid, buildon = random.sample(locations, 1)[0]
  x = True if random.random() > 0.5 else False
  l = True if random.random() > 0.5 else False
  for bid, (bx, by, bz) in locations:
    if x:
      if l:
        if bx > buildon[0] and bz == buildon[2] and by == buildon[1]:
          buildon = [bx, by, bz]
      else:
        if bx < buildon[0] and bz == buildon[2] and by == buildon[1]:
          buildon = [bx, by, bz]
    else:
      if l:
        if bx == buildon[0] and bz > buildon[2] and by == buildon[1]:
          buildon = [bx, by, bz]
      else:
        if bx == buildon[0] and bz < buildon[2] and by == buildon[1]:
          buildon = [bx, by, bz]

  if x:
    if l:
      v =  [buildon[0] + 0.1666, buildon[1], buildon[2]]
    else:
      v = [buildon[0] - 0.1666, buildon[1], buildon[2]]
  else:
    if l:
      v =  [buildon[0], buildon[1], buildon[2] + 0.1666]
    else:
      v =  [buildon[0], buildon[1], buildon[2] - 0.1666]

  # Edge of table or intersection?   Try again
  if abs(v[0]) > 0.8 or abs(v[2]) > 0.8 or intersection(v, locations):
    return walllocation(locations, block)
  return v

def convert(locs):
  locs.sort()
  locations = []
  for bid, (x, y, z) in locs:
    locations.append({"id": bid, "position": "%f,%f,%f" % (x, y, z)})
  return locations


def top(B, locs):
  tB = copy.deepcopy(B)
  for bid, (bx, by, bz) in locs:
    if tB[1][0] == bx and tB[1][2] == bz and tB[1][1] < by:
      tB = copy.deepcopy((bid, [bx, by, bz]))
  return tB


def equal(a, b):
  return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def movements(locs):
  # 20 Movements
  actions = [copy.deepcopy(locs)]
  while len(actions) <= 10:
    newlocs = copy.deepcopy(actions[len(actions)-1])
    # Find a block to move
    block = top(random.sample(newlocs, 1)[0], newlocs)
    createTower = True if random.random() < 0.1 else False
    if createTower:
      loc = towerlocation(newlocs, block[0])
    else:
      loc = randomlocation(newlocs, block[0])
    if not equal(block[1], loc):
      newlocs.remove(block)
      newlocs.append((block[0], loc))
      actions.append(newlocs)
  return actions


examples = 40
random.seed(20160115)
for i in range(examples):
  configs = open("final_%d.json" % i, 'w')
  print i
  locs = []
  # Ten block configurations (?)
  # initial "block_state"
  blocks = range(1, 11)
  ops = [1,2,3]
  while len(blocks) > 0:
    block = random.sample(blocks, 1)[0]
    blocks.remove(block)
    op = random.sample(ops,1)[0]
    if op == 1 and len(locs) > 0:
      locs.append((block, towerlocation(locs)))
    elif op == 2 and len(locs) > 0:
      locs.append((block, walllocation(locs)))
    else:
      locs.append((block, randomlocation(locs)))
  # Unravel
  moves = movements(locs)
  moves.reverse()
  locations = []
  for actions in moves:
    locations.append({"block_state": convert(actions)})
  configs.write(json.dumps({"block_states": locations, "block_meta": shape["block_meta"]}))
  configs.close()
