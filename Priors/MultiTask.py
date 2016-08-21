import os,random,sys,gzip
from PIL import Image
sys.path.append(".")

from TFLibraries.Embeddings import Embedding
from TFLibraries.Layer import Layers
from TFLibraries.Ops import *
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
Layer = Layers()

""" Parameters """
random.seed(20160408)
batch_size = 512
maxlength = 40
filters = int(sys.argv[1])
hiddendim = 100
num_epochs = 12


""" Helper Functions """

""" Create Batches for Training """
indices = []
def gen_batch_L(size, Wi, U, L, Wj):
  """
  Generate a batch for training with language data
  """
  global indices
  if len(indices) < size:
    # Randomly reorder the data
    v = range(len(Wi))
    random.shuffle(v)
    indices.extend(v)
  r = indices[:size]
  indices = indices[size:]
  return Wi[r], Wj[r], U[r],  L[r]
  ## Create zeros for world experiment
  ##Z = np.zeros((size,18,18,20))
  #return Z, Wj[r], U[r],  L[r]
  ## Create zeros for language experiment
  #Z = np.zeros((size,40))
  #return Wi[r], Wj[r], Z,  L[r]

Bindices = []
def gen_batch_B(size, Wi, U, L, Wj):
  """
  Generate a batch for training with unlabeled priors data
  """
  global Bindices
  if len(Bindices) < size:
    # Randomly reorder the data
    v = range(len(Wi))
    random.shuffle(v)
    Bindices.extend(v)
  r = Bindices[:size]
  Bindices = Bindices[size:]
  return Wi[r], Wj[r], U[r], L[r]

""" Evaluation Functions """
def eval(sess, DWi, DU, Dlens, DWj, keep_predictions=False):
  """
  Runs eval on dev/test data with the option to return predictions or performance
  """
  global batch_size
  predictions = []
  for i in range(len(DWi)/batch_size):
    batch_range = range(batch_size*i,batch_size*(i+1))
    wi = DWi[batch_range]
    wj = DWj[batch_range]
    U = DU[batch_range]
    lens = Dlens[batch_range]
    #feed_dict = {cur_world: wi, next_world: wj, inputs: U}
    feed_dict = {cur_world: wi, next_world: wj, inputs: U, lengths: lens}
    if keep_predictions:
      predictions.extend(sess.run(tf.argmax(logits,1), feed_dict))
    else:
      predictions.extend(sess.run(correct_prediction, feed_dict))

  ## Grab the extras
  last_chunk = batch_size*(i+1)
  gap = batch_size - (len(DWi) - last_chunk)
  wi = np.pad(DWi[last_chunk:], ((0,gap),(0,0), (0,0), (0,0)), mode='constant', constant_values=0)
  wj = np.pad(DWj[last_chunk:], ((0,gap),(0,0)), mode='constant', constant_values=0)
  U = np.pad(DU[last_chunk:], ((0,gap),(0,0)), mode='constant', constant_values=0)
  lens = np.pad(Dlens[last_chunk:], ((0,gap)), mode='constant', constant_values=0)
  feed_dict = {cur_world: wi, next_world: wj, inputs: U, lengths: lens}
  if keep_predictions:
    predictions.extend(sess.run(tf.argmax(logits,1), feed_dict)[:batch_size - gap])
    return predictions
  else:
    predictions.extend(sess.run(correct_prediction, feed_dict)[:batch_size - gap])
    return 100.0*sum(predictions)/len(predictions)


def real_eval(sess, DWi, DU, Dlens, DWj):
  global real_dev_id, real_dev
  # These are 1-hot representations
  predictions = eval(sess, DWi, DU, Dlens, DWj, keep_predictions=True)
  # convert predictions to (idx,idy)
  idx_idy_pairs = [(p/18, p%18) for p in predictions]
  gold_idx_idy_pairs = [(p/18, p%18) for p in np.argmax(DWj, axis=1)]
  # convert to real values
  xy_pairs = [((x - 8)*0.1528, (y - 8)*-0.1528) for (x,y) in idx_idy_pairs]
  gold_xy_pairs = [((x - 8)*0.1528, (y - 8)*-0.1528) for (x,y) in gold_idx_idy_pairs]

  # Evaluate
  Gold_locs = []
  P = []
  for i in range(len(real_dev_id)):
    P.append(block_distance([xy_pairs[i][0], 0.1, xy_pairs[i][1]],
                            real_dev[i][3*real_dev_id[i]:3*(real_dev_id[i]+1)]))
    Gold_locs.append(real_dev[i][3*real_dev_id[i]:3*(real_dev_id[i]+1)])
  P.sort()
  G = []
  for i in range(len(real_dev_id)):
    G.append(block_distance([gold_xy_pairs[i][0], 0.1, gold_xy_pairs[i][1]],
                            real_dev[i][3*real_dev_id[i]:3*(real_dev_id[i]+1)]))
  G.sort()
  #print gold_xy_pairs
  #print Gold_locs
  return (sum(P)/len(P), P[len(P)/2], sum(G)/len(G), G[len(G)/2])


def create(name, before, after, matrix, dim):
  """
    Creates an image of the confidences
  """
  img = Image.new('RGB', (dim,dim), "black")
  pixels = img.load()
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      pixels[i,j] = (int(255*before[i][j]),
                     int(255*matrix[i][j]),
                     int(255*after[i][j]))
  img.save(name)
  #img.show()

""" Data processing """

def process(U, maxlength=40, vocabsize=10000):
  """
  Need to return:  Utterenaces, lengths, vocabsize
  """
  max_vocab = 0
  lengths = []
  filtered = np.zeros(shape=[len(U), maxlength], dtype=np.int32)
  for i in range(len(U)):
    utterance = U[i]
    lengths.append(len(utterance))
    for j in range(min(len(utterance), maxlength)):
      if utterance[j] > vocabsize:
        filtered[i][j] = 1
      else:
        filtered[i][j] = utterance[j]
    max_vocab = max(max_vocab, max(utterance))
  return filtered, np.array(lengths, dtype=np.int32), max_vocab


""" Read Data """

Directory = '/home/ybisk/GroundedLanguage'
TrainData = 'Priors/Train.expL1.2.LangAndBlank.20.npz'
EvalData = 'Priors/Dev.expL1.2.Lang.20.npz'
RawEval = 'Priors/WithText/Dev.mat.gz'
#EvalData = 'Priors/Test.Lang.20.npz'
#RawEval = 'Priors/WithText/Test.mat.gz'

os.chdir(Directory)
print("Running from ", os.getcwd())
## Regular + Blank (B)
Data = np.load(TrainData)
Wi, U, Wj = Data["Lang_Wi"], Data["Lang_U"], Data["Lang_Wj"]
BWi, BU, BWj = Data["Blank_Wi"], Data["Blank_U"], Data["Blank_Wj"]
U, lens, vocabsize = process(U, maxlength)
BU, Blens, Bvocabsize = process(BU, maxlength)
## Labeled
Eval = np.load(EvalData)
DWi, DU, DWj = Eval["Lang_Wi"], Eval["Lang_U"], Eval["Lang_Wj"]
DU, Dlens, vocabsize = process(DU, maxlength, vocabsize)

# Read in the actual eval (x,y,z) data
F = gzip.open(RawEval, 'r')
real_dev = np.array([map(float, line.split()[:120]) for line in F])
# item with biggest change
real_dev_id = np.argmax(np.abs(real_dev[:,:60] - real_dev[:,60:]), axis=1)/3

""" Model Definition """

## Inputs        #[batch, height, width, depth]
cur_world = tf.placeholder(tf.float32, [batch_size, 18, 18, 20], name="CurWorld")
next_world = tf.placeholder(tf.float32, [batch_size, 324], name="NextWorld")
## Language
inputs = tf.placeholder(tf.int32, [batch_size, maxlength], name="Utterance")
lengths = tf.placeholder(tf.int32, [batch_size], name="Lengths")

## weights && Convolutions
W = {
  'cl1': Layer.convW([3, 3, 20, filters]),
  'cl2': Layer.convW([3, 3, filters, filters]),
  'cl3': Layer.convW([3, 3, filters, filters]),
  'out': Layer.W(12*12*filters + 2*hiddendim, 324)
}

B = {
  'cb1': Layer.b(filters, init='Normal'),
  'cb2': Layer.b(filters, init='Normal'),
  'cb3': Layer.b(filters, init='Normal'),
  'out': Layer.b(324)
}

# Define embeddings matrix
embeddings = Embedding(vocabsize, one_hot=False, embedding_size=hiddendim)

# RNN
dropout = 0.75
lstm = tf.nn.rnn_cell.LSTMCell(hiddendim,
               initializer=tf.contrib.layers.xavier_initializer(seed=20160501))
lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout)

# Encode from 18x18 to 12x12
l1 = conv2d('l1', cur_world, W['cl1'], B['cb1'], padding='VALID') # -> 16x16
l2 = conv2d('l2', l1, W['cl2'], B['cb2'], padding='VALID')        # -> 14x14
l3 = conv2d('l3', l2, W['cl3'], B['cb3'], padding='VALID')        # -> 12x12

outputs, fstate = tf.nn.dynamic_rnn(lstm, embeddings.lookup(inputs), 
                                    sequence_length=lengths,
                                    dtype=tf.float32)

# Concatenate RNN output to CNN representation
logits = tf.matmul(
          tf.concat(1, [fstate,
            tf.reshape(l3, [batch_size,12*12*filters])]),
        W['out']) + B['out']
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(next_world,1))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,next_world))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
total_loss = 0.0

ratio = 1.0

def run_step((batch_Wi, batch_Wj, batch_U, batch_L)):
  feed_dict = {cur_world: batch_Wi, next_world: batch_Wj,
               inputs: batch_U, lengths: batch_L}
  loss_val, t_op = sess.run([loss, train_op], feed_dict)
  return loss_val

discrete = []
real = []

for epoch in range(num_epochs):
  for step in range(BWi.shape[0]/batch_size):    ## Does not make use of full prior data this way
    total_loss += run_step(gen_batch_B(batch_size, BWi, BU, Blens, BWj))
  discrete.append(eval(sess, DWi, DU, Dlens, DWj))
  real.append(real_eval(sess, DWi, DU, Dlens, DWj))
  print("Iter %3d  Ratio %-6.4f  Loss %-10f   Eval  %-6.3f  %5.3f  %5.3f  G: %5.3f %5.3f" %
       (epoch, ratio, total_loss, discrete[-1], real[-1][0], real[-1][1], real[-1][2], real[-1][3]))
  total_loss = 0
    #ratio = 1.0 - epoch/25.0
print "Convereged on Priors"


for epoch in range(num_epochs):
  for step in range(Wi.shape[0]/batch_size):
    total_loss += run_step(gen_batch_L(batch_size, Wi, U, lens, Wj))
  discrete.append(eval(sess, DWi, DU, Dlens, DWj))
  real.append(real_eval(sess, DWi, DU, Dlens, DWj))
  print("Iter %3d  Ratio %-6.4f  Loss %-10f   Eval  %-6.3f  %5.3f  %5.3f  G: %5.3f %5.3f" %
        (epoch, ratio, total_loss, discrete[-1], real[-1][0], real[-1][1], real[-1][2], real[-1][3]))
  total_loss = 0
print "Converged on Language"

print "Grid v XYZ correlation: %5.3f %5.3f" % (
  pearson([r[0] for r in real], discrete), pearson([r[1] for r in real], discrete))

"""
 Print images showing the predictions of the model
"""

def collapse(F):
  M = np.zeros((18,18))
  for i in range(18):
    for j in range(18):
      if np.amax(F[i][j]) > 0:
          M[i][j] = 1
  return M

s = np.array(range(batch_size))
feed_dict = {cur_world: DWi[s], next_world: DWj[s],
             inputs: DU[s], lengths: Dlens[s]}
final = sess.run(logits, feed_dict)
for i in range(batch_size):
  # Show the final prediction confidences
  create("P_%d_%d.bmp" % (epoch, i),
      collapse(DWi[s][i]),
      np.reshape(DWj[s], (batch_size,18,18))[i],
      np.reshape(final, (batch_size,18,18))[i], 18)


