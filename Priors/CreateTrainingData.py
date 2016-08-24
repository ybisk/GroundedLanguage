import numpy as np
from PIL import Image
import os,gzip,math,sys

dimensions = 64
offset = dimensions/2 - 1
block_size = 0.1528
space_size = 3.0
unit_size = space_size / dimensions

## Read Worlds  a = numpy.loadtxt('data.txt')
def read(s):
  D = []
  for line in gzip.open(s):
    line = line.strip().split()
    Wj = [float(a) for a in line[:60]]
    Wi = [float(a) for a in line[60:120]]
    if len(line) > 120:
      U = [int(a) for a in line[120:]]
    else:
      U = [1]   # Single unk, maybe should be a special token?
    D.append((Wi, U, Wj))
  print "Read ", len(D), " records"
  return D

# Preprocess a world
def preprocess(X):
  while X[len(X)-1] == -1 and X[len(X)-2] == -1 and X[len(X)-3] == -1:
    X = X[:len(X)-3]
  X = np.reshape(X, (len(X)/3,3))
  X /= [1,1,-1]
  X += space_size/2
  ## This block covers which pixels
  Labels = []
  MidPoints = []
  for i in range(len(X)):
    row = X[i]
    LL = (int(math.floor((row[0] - block_size/2)/unit_size)),
          int(math.ceil((row[2] - block_size/2)/unit_size)))
    UR = (int(math.floor((row[0] + block_size/2)/unit_size)),
          int(math.ceil((row[2] + block_size/2)/unit_size)))
    MidPoints.append((int(row[0]/unit_size), int(row[2]/unit_size)))
    for r in range(LL[0], UR[0] + 1):
      for c in range(LL[1], UR[1] + 1):
        Labels.append((i, r,c))
  return Labels, MidPoints

# Convert to filter grid
def coordToGrid(X, Y=None):
  # Preprocess X
  X, _ = preprocess(X)
  # Preprocess Y
  if Y != None:
    Y, _ = preprocess(Y)
  G = np.zeros(shape=[dimensions,dimensions,20], dtype=np.float32)
  for (idx, row, col) in X:
    G[row][col][idx] = 1
    if Y != None:
      G[row][col][idx] += 0.5
  return G

def create1Hot(X, Y):
  X, _ = preprocess(X)
  Y, My = preprocess(Y)
  H = np.zeros(shape=[dimensions,dimensions,1], dtype=np.float64)
  block_id = -1
  for (idx, row, col) in set(Y).difference(set(X)):
    H[row][col] += 1
    block_id = idx
  if block_id == -1:
    print "Houston, we have a problem"
  (idx, idy) = My[block_id]
  for i in range(dimensions):
    for j in range(dimensions):
      L1 = abs(i-idx) + abs(j - idy)
      if L1 < dimensions/10:
        H[i][j] += 1.0/math.exp(L1)
  H = np.reshape(H, [dimensions*dimensions])
  if np.sum(H) == 0.0:
    print "Error", X == Y, d
    sys.exit()
  H = H/(1.0*np.sum(H))
  return H

def grid(X):
  Wi = np.zeros(shape=[len(X), dimensions, dimensions, 20], dtype=np.int32)
  U = []
  Wj = np.zeros(shape=[len(X), dimensions*dimensions], dtype=np.float32)
  for j in range(len(X)):
    World_i, Utterance, World_j = X[j]
    # Grid worlds
    Wi[j] = coordToGrid(World_i)
    Wj[j] = create1Hot(World_i, World_j)
    U.append(Utterance)
  print "Converted to grids"
  return Wi, U, Wj


#os.chdir('/Users/ybisk/Dropbox/GroundedLanguage_Clean')
os.chdir('/home/ybisk/GroundedLanguage')
print("Running from ", os.getcwd())

Blank = read("Priors/extra.next.current.20.mat.gz")
Lang = read("Priors/WithText/Train.mat.gz")

Wi, U, Wj = grid(Lang)
BWi, BU, BWj = grid(Blank)
np.savez("Train.%d.L1.LangAndBlank.20" % dimensions, Lang_Wi=Wi, Lang_Wj=Wj, Lang_U=U,
                                Blank_Wi=BWi, Blank_Wj=BWj, Blank_U=BU)
Lang = read("Priors/WithText/Dev.mat.gz")
Wi, U, Wj = grid(Lang)
np.savez("Dev.%d.L1.Lang.20" % dimensions, Lang_Wi=Wi, Lang_Wj=Wj, Lang_U=U)
