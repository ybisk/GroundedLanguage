import numpy as np

class SparseFiles:

  ## Create sparse numpy arrays
  def sparsify(self, data, V, length, off, prediction=0, concat=True):
    if concat:
      D = np.zeros(shape=[len(data), length * V], dtype=np.float32)
    else:
      D = np.zeros(shape=[len(data), length,  V], dtype=np.float32)
    L = np.zeros(shape=[len(data), 20], dtype=np.float32)
    lens = []
    for j in range(len(data)):
      sentence = data[j]
      for i in range(min(length,len(sentence)) - off):
        if concat:
          D[j][V * i + sentence[off + i]] = 1
        else:
          D[j][i][sentence[off + i]] = 1
      for i in range(len(sentence), length):
        # Padding
        if concat:
          D[j][V * i] = 1
        else:
          D[j][i][0] = 1
      L[j][sentence[prediction]] = 1  # 0 Source, 1 Reference, 2 Direction
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
    return T, np.array(l), v
