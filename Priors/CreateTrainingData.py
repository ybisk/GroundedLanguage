import numpy as np
import os,gzip

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
  X /= [0.1528,1,-0.1528]
  X += 8
  X = [[int(round(v)) for v in row] for row in X]
  return X

# Convert to filter grid
def coordToGrid(X, Y=None):
  # Preprocess X
  X = preprocess(X)
  # Preprocess Y
  if Y != None:
    Y = preprocess(Y)

  G = np.zeros(shape=[18,18,20], dtype=np.float32)
  for i in range(len(X)):
    G[X[i][0]][X[i][2]][i] = 1
    if Y != None:
      G[Y[i][0]][Y[i][2]][i] += 0.5
  return G

def create1Hot(X, Y):
  X = preprocess(X)
  Y = preprocess(Y)
  G = np.zeros(shape=[18,18,1], dtype=np.int32)
  H = np.zeros(shape=[18,18,1], dtype=np.int32)
  for i in range(len(Y)):
    G[X[i][0]][X[i][2]] = 1
    H[Y[i][0]][Y[i][2]] = 1
  cur_world = np.reshape(G, [324])
  next_world = np.reshape(H, [324])
  d = [a_i - b_i for a_i, b_i in zip(next_world, cur_world)]
  return d.index(max(d))

def grid(X):
  Wi = np.zeros(shape=[len(X), 18, 18, 20], dtype=np.int32)
  U = []
  Wj = np.zeros(shape=[len(X), 324], dtype=np.int32)
  for j in range(len(X)):
    World_i, Utterance, World_j = X[j]
    # Grid worlds
    Wi[j] = coordToGrid(World_i)
    Wj[j][create1Hot(World_i, World_j)] = 1
    U.append(Utterance)
  print "Converted to grids"
  return Wi, U, Wj


#os.chdir('/Users/ybisk/Dropbox/GroundedLanguage_Clean')
os.chdir('/home/ybisk/GroundedLanguage')
print("Running from ", os.getcwd())

Train = False
if Train:
  Blank = read("Priors/extra.next.current.20.mat.gz")
  Lang = read("Priors/WithText/Train.mat.gz")

  Wi, U, Wj = grid(Lang)
  BWi, BU, BWj = grid(Blank)
  np.savez("Train.LangAndBlank.20", Lang_Wi=Wi, Lang_Wj=Wj, Lang_U=U,
                                  Blank_Wi=BWi, Blank_Wj=BWj, Blank_U=BU)
else:
  Lang = read("Priors/WithText/Dev.mat.gz")
  Wi, U, Wj = grid(Lang)
  np.savez("Eval.Lang.20", Lang_Wi=Wi, Lang_Wj=Wj, Lang_U=U)
