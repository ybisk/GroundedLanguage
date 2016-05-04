import numpy as np

class SparseFiles:

  def __init__(self, length, off, labelspace=20, prediction=0):
    self.vocab = -1
    self.length = length
    self.offset = off
    self.label_space = labelspace
    self.prediction = prediction

  ## Create sparse numpy arrays
  def matrix(self, data):
    D = np.zeros(shape=[len(data), self.length], dtype=np.int32)
    L = np.zeros(shape=[len(data), self.label_space], dtype=np.float32)
    lens = []
    for j in range(len(data)):
      sentence = data[j]
      for i in range(min(self.length,len(sentence)) - self.offset):
          D[j][i] = sentence[self.offset + i]
      for i in range(len(sentence), self.length):   # Unk Padding
          D[j][i] = 1
      L[j][sentence[self.prediction]] = 1  # 0 Source, 1 Reference, 2 Direction
    print(np.shape(D), np.shape(L))
    return D, L

  ## Read sparse arrays
  def read(self, s):
    T = []
    v = 0
    l = []
    for line in open(s):
      t = [int(a) for a in line.strip().split()]
      l.append(len(t) + 1)
      v = max(v, max(t))
      T.append(t)
    if self.vocab == -1:
      v += 1
      self.vocab = v
    return T, np.array(l), v
