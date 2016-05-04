import os,random,sys
sys.path.append(".")

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from TFLibraries.Layer import Layers
from TFLibraries.Train import Training
from TFLibraries.Sparse import SparseFiles
from TFLibraries.Embeddings import Embedding
Layer = Layers()

random.seed(20160408)

indices = []
def generate_batch(size, data, labels, lengths):
  global indices
  if len(indices) < size:
    indices.extend(range(data.shape[0]))
  # Random indices
  r = random.sample(indices, size)
  indices = filter(lambda a: a not in r, indices)
  return data[r], labels[r], lengths[r]

## Read Training/Dev/Test data
os.chdir('/home/ybisk/GroundedLanguage')
print("Running from ", os.getcwd())
maxlength = 80
offset = 3
labelspace = 9
Sparse = SparseFiles(maxlength, offset, labelspace=labelspace, prediction=2)
train, train_lens, vocabsize = Sparse.read("JSONReader/data/2016-NAACL/SRD/Train.mat")
dev, dev_lens, _             = Sparse.read("JSONReader/data/2016-NAACL/SRD/Dev.mat")
test, test_lens, _           = Sparse.read("JSONReader/data/2016-NAACL/SRD/Test.mat")

## Create sparse arrays
training, training_labels       = Sparse.matrix(train)
development, development_labels = Sparse.matrix(dev)
testing, testing_labels         = Sparse.matrix(test)

## TODO:
## MutiCellLSTM

batch_size = 128
hiddendim = 256
embeddingdim = 100
onehot = False
graph = tf.Graph()
dropout=0.5

# Define embeddings matrix
embeddings = Embedding(vocabsize, one_hot=onehot, embedding_size=embeddingdim)

# Input -> LSTM -> Outstate
inputs = tf.placeholder(tf.int32, [batch_size, maxlength])
labels = tf.placeholder(tf.float32, [batch_size, labelspace])
lengths = tf.placeholder(tf.int32, [batch_size])


# RNN
lstm = tf.nn.rnn_cell.LSTMCell(hiddendim,
                               initializer=tf.contrib.layers.xavier_initializer(seed=20160501))
lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout)

# Prediction
output_layer = Layer.W(2*hiddendim, labelspace, 'Output')
output_bias  = Layer.b(labelspace, 'OutputBias')

outputs, fstate = tf.nn.dynamic_rnn(lstm, embeddings.lookup(inputs), 
                                    sequence_length=lengths, 
                                    dtype=tf.float32)
logits = tf.matmul(fstate, output_layer) + output_bias
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

## Learning ##
## Optimizer Adam/RMSPropOptimizer   tf.train.AdamOptimizer()
optimizer = tf.train.AdamOptimizer()
## Gradient Clipping:
##tvars = tf.trainable_variables()
##grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
##train_op = optimizer.apply_gradients(zip(grads, tvars))
train_op = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))

# Add this to session to see cpu/gpu placement: 
# config=tf.ConfigProto(log_device_placement=True)
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  # Model params
  Trainer = Training(sess, correct_prediction, train_op, loss, 
                     inputs, labels, lengths, batch_size)
  # Run training
  Trainer.train(training, training_labels, development, development_labels, 
                generate_batch, train_lens, dev_lens)
