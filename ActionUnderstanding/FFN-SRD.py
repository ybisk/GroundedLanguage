import random,sys,os
sys.path.append(".")

import tensorflow as tf
from TFLibraries.Layer import Layers
from TFLibraries.Train import Training
from TFLibraries.Sparse import SparseFiles
from TFLibraries.Embeddings import Embedding
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
maxlength = 80
offset = 3
labelspace = 9
Sparse = SparseFiles(maxlength, offset, labelspace=labelspace, prediction=2)
train, _, vocabsize = Sparse.read("JSONReader/data/2016-NAACL/SRD/Train.mat")
dev, _, _           = Sparse.read("JSONReader/data/2016-NAACL/SRD/Dev.mat")
test, _, _          = Sparse.read("JSONReader/data/2016-NAACL/SRD/Test.mat")

## Create sparse arrays
training, training_labels       = Sparse.matrix(train)
development, development_labels = Sparse.matrix(dev)
testing, testing_labels         = Sparse.matrix(test)

batch_size = 128
hiddendim = 100
embeddingdim = 100
graph = tf.Graph()
onehot = False
inputdim = maxlength*vocabsize if onehot else maxlength*embeddingdim

# Define embeddings matrix
embeddings = Embedding(vocabsize, one_hot=onehot, embedding_size=embeddingdim)
# Input data.
dataset = tf.placeholder(tf.int32, shape=[batch_size, maxlength], name='Train')
labels = tf.placeholder(tf.float32, shape=[batch_size, labelspace], name='Label')
# Model
hidden_layer = Layer.W(inputdim, hiddendim, 'Hidden')
hidden_bias  = Layer.b(hiddendim, 'HiddenBias')
# Prediction
output_layer = Layer.W(hiddendim, labelspace, 'Output')
output_bias  = Layer.b(labelspace, 'OutputBias')

embedded = tf.reshape(embeddings.lookup(dataset), [batch_size,inputdim])
forward = tf.nn.relu(tf.matmul(embedded, hidden_layer) + hidden_bias)
dropout = tf.nn.dropout(forward, 0.5)
logits = tf.matmul(dropout, output_layer) + output_bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
train_op = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  # Model params
  Trainer = Training(sess, correct_prediction, train_op, loss, dataset, labels)
  # Run training
  Trainer.train(training, training_labels, development, development_labels, generate_batch)
