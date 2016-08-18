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

random.seed(20160408)
batch_size = 512

indices = []
def gen_batch_L(size, Wi, U, L, Wj):
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
    global Bindices
    if len(Bindices) < size:
      # Randomly reorder the data
      v = range(len(Wi))
      random.shuffle(v)
      Bindices.extend(v)
    r = Bindices[:size]
    Bindices = Bindices[size:]
    return Wi[r], Wj[r], U[r], L[r]


def eval(sess, batch_size, DWi, DU, Dlens, DWj, predictions=False):
  predictions = []
  for i in range(len(DWi)/batch_size):
    batch_range = range(batch_size*i,batch_size*(i+1))
    wi = DWi[batch_range]
    wj = DWj[batch_range]
    U = DU[batch_range]
    lens = Dlens[batch_range]
    #feed_dict = {cur_world: wi, next_world: wj, inputs: U}
    feed_dict = {cur_world: wi, next_world: wj, inputs: U, lengths: lens}
    if predictions:
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
  if predictions:
    predictions.extend(sess.run(tf.argmax(logits,1), feed_dict)[:batch_size - gap])
  else:
    predictions.extend(sess.run(correct_prediction, feed_dict)[:batch_size - gap])
  return 100.0*sum(predictions)/len(predictions)

def create(name, before, after, matrix, dim):
  img = Image.new('RGB', (dim,dim), "black")
  pixels = img.load()
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      pixels[i,j] = (int(255 - 255*before[i][j][1]),
                     int(255 - 255*matrix[i][j]),
                     int(255 - 255*after[i][j]))
  img.save(name)
  #img.show()

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

##### Read Data #####
os.chdir('/home/ybisk/GroundedLanguage')
print("Running from ", os.getcwd())
## Regular + Blank (B)
Data = np.load("Priors/Train.LangAndBlank.20.npz")
Wi, U, Wj = Data["Lang_Wi"], Data["Lang_U"], Data["Lang_Wj"]
BWi, BU, BWj = Data["Blank_Wi"], Data["Blank_U"], Data["Blank_Wj"]
maxlength = 40
U, lens, vocabsize = process(U, maxlength)
BU, Blens, Bvocabsize = process(BU, maxlength)
## Labeled
Eval = np.load("Priors/Eval.Lang.20.npz")
DWi, DU, DWj = Eval["Lang_Wi"], Eval["Lang_U"], Eval["Lang_Wj"]
DU, Dlens, vocabsize = process(DU, maxlength, vocabsize)


##### Model Definition #####

## Inputs        #[batch, height, width, depth]
cur_world = tf.placeholder(tf.float32, [batch_size, 18, 18, 20], name="CurWorld")
next_world = tf.placeholder(tf.float32, [batch_size, 324], name="NextWorld")
## Language
inputs = tf.placeholder(tf.int32, [batch_size, maxlength], name="Utterance")
lengths = tf.placeholder(tf.int32, [batch_size], name="Lengths")

## weights && Convolutions
filters = int(sys.argv[1])
hiddendim = 100
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
num_epochs = 25
total_loss = 0.0
s = range(batch_size)

ratio = 1.0

for epoch in range(num_epochs):
    for step in range(BWi.shape[0]/batch_size):    ## Does not make use of full prior data this way
      r = random.random()
      if r > ratio:
        batch_Wi, batch_Wj, batch_U, batch_L = gen_batch_L(batch_size, Wi, U, lens, Wj)
      else:
        batch_Wi, batch_Wj, batch_U, batch_L = gen_batch_B(batch_size, BWi, BU, Blens, BWj)
      feed_dict = {cur_world: batch_Wi, next_world: batch_Wj,
                   inputs: batch_U, lengths: batch_L}
      loss_val, t_op = sess.run([loss, train_op], feed_dict)
      total_loss += loss_val
    print("Iter %3d  Ratio %-6.4f  Loss %-10f   Dev  %-5.3f" %
         (epoch, ratio, total_loss, eval(sess, batch_size, DWi, DU, Dlens, DWj)))
    total_loss = 0
    ratio = 1.0 - epoch/25.0

#for i in range(512):
#  # Show the final prediction confidences
#  create("P_%d_%d.bmp" % (epoch, i), 
#      Train[0][s][i], 
#      np.reshape(Train[1][s], (batch_size,18,18))[i], 
#      np.reshape(final, (batch_size,18,18))[i], 18)
