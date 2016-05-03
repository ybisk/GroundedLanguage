import random,sys,os
sys.path.append(".")

import tensorflow as tf
from TFLibraries.Layer import Layers
from TFLibraries.Train import Training
from TFLibraries.Sparse import SparseFiles
Sparse = SparseFiles()
Layer = Layers()

random.seed(20160408)

indices = []
def generate_batch(size, data, labels):
  global indices
  if len(indices) < size:
    indices.extend(range(data.shape[0]))
  r = random.sample(indices, size)
  indices = filter(lambda a: a not in r, indices)
  # Randomly reorder the data
  return data[r], labels[r]

## Read Training/Dev/Test data
os.chdir('/home/ybisk/GroundedLanguage')
print("Running from ", os.getcwd())
train, length, vocabsize = Sparse.read("JSONReader/data/2016-NAACL/SRD/Train.mat")
dev, _, _                = Sparse.read("JSONReader/data/2016-NAACL/SRD/Dev.mat")
test, _, _               = Sparse.read("JSONReader/data/2016-NAACL/SRD/Test.mat")
offset = 3

## Create sparse arrays
training, training_labels       = Sparse.sparsify(train, vocabsize, length, offset)
development, development_labels = Sparse.sparsify(dev, vocabsize, length, offset)
testing, testing_labels         = Sparse.sparsify(test, vocabsize, length, offset)

batch_size = 128
hiddendim = 100
outputdim = 20
graph = tf.Graph()

# Input data.
dataset = tf.placeholder(tf.float32, shape=[batch_size, length*vocabsize], name='Train')
labels = tf.placeholder(tf.float32, shape=[batch_size, outputdim], name='Label')
# Model
hidden_layer = Layer.W(length*vocabsize, hiddendim, 'Hidden')
hidden_bias  = Layer.b(hiddendim, 'HiddenBias')
# Prediction
output_layer = Layer.W(hiddendim, outputdim, 'Output')
output_bias  = Layer.b(outputdim, 'OutputBias')

forward = tf.nn.relu(tf.matmul(dataset, hidden_layer) + hidden_bias)
dropout = tf.nn.dropout(forward, 0.5)
logits = tf.matmul(dropout, output_layer) + output_bias

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
train_op = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  # Model params
  Trainer = Training(sess, correct_prediction, train_op, loss, dataset, labels)
  # Run training
  Trainer.train(training, training_labels, development, development_labels, generate_batch)
