import os
import copy
import random
import gzip
import sys
import json

import editdistance
from nltk.tokenize import TreebankWordTokenizer

# Input:  {"world":
#           [0.8062, 0.1, -0.5769,
#           0.4595, 0.1, -0.4644,
#           -0.5604, 0.1, 0.4731,
#           -0.6564, 0.1, 0.764,
#           0, 0.1, 0,
#           0, 0.1, -0.1667,
#           0.6544, 0.1, -0.919,
#           -0.4754, 0.1, -0.2636,
#           -0.3238, 0.1, 0.0131,
#           0.1667, 0.1, 0.3333,
#           0.1667, 0.1, 0.1667,
#           0.1667, 0.1, 0,
#           0.1667, 0.1, -0.1667,
#           0.3333, 0.1, 0.6667,
#           0.3333, 0.1, 0.5,
#           0.3333, 0.1, 0.3333,
#           0.5, 0.1, 0.6667],
#         "ex": 0,
#         "text": "move the adidas block directly diagonally left and below the heineken block .  "}
# Output: {"loc": [-0.1667, 0.1, -0.3333],
#          "id": 1,
#          "ex": 0}

def genWorld(D):
  """
  Flatten a dictionary
  :param D:
  :return:
  """
  l = []
  for i in range(len(D)):
    l.extend(D[i])
  return l

brands = [
  'adidas', 'bmw', 'burger king', 'coca cola', 'esso',
  'heineken', 'hp', 'mcdonalds', 'mercedes benz', 'nvidia',
  'pepsi', 'shell', 'sri', 'starbucks', 'stella artois',
  'target', 'texaco', 'toyota', 'twitter', 'ups']

Multiplier = 10

####  Read training data ####
Input = []
Output = []

AutoInput = []
AutoOutput = []

for line in gzip.open(sys.argv[1],'r'):
  Input.append(json.loads(line))
  AutoInput.append(json.loads(line))

for line in gzip.open(sys.argv[2],'r'):
  Output.append(json.loads(line))
  AutoOutput.append(json.loads(line))

for i in range(len(Input)):
  inp = Input[i]
  out = Output[i]

  ## Create dictionary of block locations ##
  world = {}
  for index in range(len(inp["world"])/3):
    world[index] = [inp["world"][index*3], inp["world"][index*3 + 1], inp["world"][index*3 + 2]]

  ## Get the set of (explicitely) referenced blocks
  words = TreebankWordTokenizer().tokenize(inp["text"])
  blocks = set()
  blocks.add(out["id"] - 1)
  for brand in brands:
    brandparts = brand.split()
    for part in brandparts:
      for word in words:
        if editdistance.eval(part, word) < 2:
          blocks.add(brands.index(brand))

  ## Assumed invariant to the instruction ##
  invariant = set(range(0, len(world))).difference(blocks)

  for m in range(Multiplier - 1):
    x,y = random.sample(invariant, 2)
    newWorld = copy.deepcopy(world)
    newWorld[x] = world[y]
    newWorld[y] = world[x]

    exid = len(AutoInput)
    AutoInput.append({"text": inp["text"], "world": genWorld(newWorld), "ex": exid})
    AutoOutput.append({"loc": out["loc"], "id": out["id"], "ex": exid})

    if len(AutoInput) % 10000 == 0:
      print len(AutoInput)


print "Final: ", len(AutoInput)
inp_file = gzip.open("Input.Auto.json.gz",'w')
out_file = gzip.open("Output.Auto.json.gz",'w')
for i in range(len(AutoInput)):
  inp_file.write(json.dumps(AutoInput[i]) + "\n")
  out_file.write(json.dumps(AutoOutput[i]) + "\n")
inp_file.close()
out_file.close()

