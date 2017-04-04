import os,random,sys
sys.path.append(".")

## Model Imports
import tensorflow as tf
tf.set_random_seed(20160905)
import numpy as np
np.set_printoptions(threshold=np.nan)
from TFLibraries.Layer import Layers
from TFLibraries.Train import Training
from TFLibraries.Sparse import SparseFiles
from TFLibraries.Embeddings import Embedding
Layer = Layers()

## Server Code
import json
import time
from flask import Flask
from flask import request
from flask import jsonify

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


os.chdir('/home/ybisk/GroundedLanguage')
print("Running from ", os.getcwd())
maxlength = 80
offset = 3
batch_size = 10
hiddendim = 256
embeddingdim = hiddendim
onehot = False
dropout=0.5

## Create Data
training = {}
training_labels = {}
training_lens = {}
development = {}
development_labels = {}
development_lens = {}
testing = {}
testing_labels = {}
testing_lens = {}
dataType = ["source","reference","direction"]
for prediction in [0,1,2]:

  ## Read Training/Dev/Test data
  labelspace = [20,20,9]
  labelspace = labelspace[prediction]
  print "Read ", dataType[prediction]
  Sparse = SparseFiles(maxlength, offset, labelspace=labelspace, prediction=prediction)
  train, train_lens, vocabsize = Sparse.read("JSONReader/data/2016-Version2/SRD/Train.mat")
  dev, dev_lens, _             = Sparse.read("JSONReader/data/2016-Version2/SRD/Dev.mat")
  test, test_lens, _           = Sparse.read("JSONReader/data/2016-Version2/SRD/Test.mat")

  training_lens[prediction] = train_lens
  development_lens[prediction] = dev_lens
  testing_lens[prediction] = test_lens

  ## Create sparse arrays
  t, t_l = Sparse.matrix(train)
  training[prediction] = t
  training_labels[prediction] = t_l
  d, d_l = Sparse.matrix(dev)
  development[prediction] = d
  development_labels[prediction] = d_l
  t, t_l = Sparse.matrix(test)
  testing[prediction] = t
  testing_labels[prediction] = t_l

# Define embeddings matrix
embeddings = {}
embeddings[0] = Embedding(vocabsize, one_hot=onehot, embedding_size=embeddingdim)
embeddings[1] = Embedding(vocabsize, one_hot=onehot, embedding_size=embeddingdim)
embeddings[2] = Embedding(vocabsize, one_hot=onehot, embedding_size=embeddingdim)

# Input -> LSTM -> Outstate
inputs = tf.placeholder(tf.int32, [batch_size, maxlength])
labels = {}
labels[0] = tf.placeholder(tf.float32, [batch_size, 20])
labels[1] = tf.placeholder(tf.float32, [batch_size, 20])
labels[2] = tf.placeholder(tf.float32, [batch_size, 9])
lengths = tf.placeholder(tf.int32, [batch_size])

multicells = 1

# RNN
lstm = {}
lstm[0] = tf.contrib.rnn.LSTMCell(hiddendim, state_is_tuple=True,
                               initializer=tf.contrib.layers.xavier_initializer(seed=20160501))
lstm[1] = tf.contrib.rnn.LSTMCell(hiddendim, state_is_tuple=True,
                               initializer=tf.contrib.layers.xavier_initializer(seed=20160501))
lstm[2] = tf.contrib.rnn.LSTMCell(hiddendim, state_is_tuple=True,
                               initializer=tf.contrib.layers.xavier_initializer(seed=20160501))
lstm[0] = tf.contrib.rnn.DropoutWrapper(lstm[0], output_keep_prob=dropout)
lstm[1] = tf.contrib.rnn.DropoutWrapper(lstm[1], output_keep_prob=dropout)
lstm[2] = tf.contrib.rnn.DropoutWrapper(lstm[2], output_keep_prob=dropout)
lstm[0] = tf.contrib.rnn.MultiRNNCell(cells=[lstm[0]] * multicells, state_is_tuple=True)
lstm[1] = tf.contrib.rnn.MultiRNNCell(cells=[lstm[1]] * multicells, state_is_tuple=True)
lstm[2] = tf.contrib.rnn.MultiRNNCell(cells=[lstm[2]] * multicells, state_is_tuple=True)


# Prediction
output_layer = {}
output_layer[0] = Layer.W(multicells*hiddendim, 20, 'Output-Sou')
output_layer[1] = Layer.W(multicells*hiddendim, 20, 'Output-Ref')
output_layer[2] = Layer.W(multicells*hiddendim, 9, 'Output-Dir')
output_bias = {}
output_bias[0]  = Layer.b(20, 'OutputBias-Sou')
output_bias[1]  = Layer.b(20, 'OutputBias-Ref')
output_bias[2]  = Layer.b(9, 'OutputBias-Dir')

outputs = {}
fstate = {}
with tf.variable_scope("lstm0"):
  outputs[0], fstate[0] = tf.nn.dynamic_rnn(lstm[0], embeddings[0].lookup(inputs),
                                      sequence_length=lengths, 
                                      dtype=tf.float32)
with tf.variable_scope("lstm1"):
  outputs[1], fstate[1] = tf.nn.dynamic_rnn(lstm[1], embeddings[1].lookup(inputs),
                                      sequence_length=lengths, 
                                      dtype=tf.float32)
with tf.variable_scope("lstm2"):
  outputs[2], fstate[2] = tf.nn.dynamic_rnn(lstm[2], embeddings[2].lookup(inputs),
                                      sequence_length=lengths, 
                                      dtype=tf.float32)
logits = {}
# prediction, layer, c/m
logits[0] = tf.matmul(tf.concat([f.h for f in fstate[0]], 1), output_layer[0]) + output_bias[0]
logits[1] = tf.matmul(tf.concat([f.h for f in fstate[1]], 1), output_layer[1]) + output_bias[1]
logits[2] = tf.matmul(tf.concat([f.h for f in fstate[2]], 1), output_layer[2]) + output_bias[2]
#logits[0] = tf.matmul(fstate[0][0], output_layer[0]) + output_bias[0]
#logits[1] = tf.matmul(fstate[1][0], output_layer[1]) + output_bias[1]
#logits[2] = tf.matmul(fstate[2][0], output_layer[2]) + output_bias[2]
loss = {}
loss[0] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[0], labels=labels[0]))
loss[1] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[1], labels=labels[1]))
loss[2] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[2], labels=labels[2]))

## Learning ##
optimizer = {}
optimizer[0] = tf.train.AdamOptimizer()
optimizer[1] = tf.train.AdamOptimizer()
optimizer[2] = tf.train.AdamOptimizer()
## Gradient Clipping:
train_op = {}
train_op[0] = optimizer[0].minimize(loss[0])
train_op[1] = optimizer[1].minimize(loss[1])
train_op[2] = optimizer[2].minimize(loss[2])
correct_prediction = {}
correct_prediction[0] = tf.equal(tf.argmax(logits[0],1), tf.argmax(labels[0],1))
correct_prediction[1] = tf.equal(tf.argmax(logits[1],1), tf.argmax(labels[1],1))
correct_prediction[2] = tf.equal(tf.argmax(logits[2],1), tf.argmax(labels[2],1))

DoTrain = False

def offset(loc, direction):
  off = 0.1666
  if direction == 6:
    return [loc[0] - off, loc[1], loc[2] - off]
  elif direction == 3:
    return [loc[0] - off, loc[1], loc[2]]
  elif direction == 0:
    return [loc[0] - off, loc[1], loc[2] + off]
  elif direction == 7:
    return [loc[0], loc[1], loc[2] - off]
  elif direction == 4:
    return [loc[0], loc[1] + off, loc[2]]
  elif direction == 1:
    return [loc[0], loc[1], loc[2] + off]
  elif direction == 8:
    return [loc[0] + off, loc[1], loc[2] - off]
  elif direction == 5:
    return [loc[0] + off, loc[1], loc[2]]
  elif direction == 2:
    return [loc[0] + off, loc[1], loc[2] + off]

def vocab(word):
  global Vocabulary
  return Vocabulary[word] if word in Vocabulary else 1

app = Flask(__name__)
@app.route("/")
def mainack():
    return "Server Up and Running"
@app.route("/query", methods=['GET', 'POST'])
def query():
  global sess
  if request.method == 'POST':
    postdat = json.loads(request.get_data().decode("utf-8"))

    # Create a batch
    utterance = [vocab(a) for a in postdat['input'].lower().strip().split()][:maxlength]
    D = np.zeros(shape=[batch_size, maxlength], dtype=np.int32)
    L = np.zeros(shape=[batch_size], dtype=np.int32)
    L[0] = len(utterance)
    for i in range(len(utterance)):
      D[0][i] = utterance[i]

    # Pass world+utterance to AI
    S = sess.run(tf.argmax(logits[0],1), feed_dict = {inputs: D, lengths: L})[0]+1
    R = sess.run(tf.argmax(logits[1],1), feed_dict = {inputs: D, lengths: L})[0]
    D = sess.run(tf.argmax(logits[2],1), feed_dict = {inputs: D, lengths: L})[0]

    if R < len(postdat['world']):
      reference = postdat['world'][R]['loc']
      new_loc = offset(reference, D)
      error = None
    else:
      error = "Can't handle"

    if S > len(postdat['world']):
      error = "Can't handle"
    
    if error != None:
      output = {"world":[{"loc": [-0.15, 0.07161243521606164, -0.15], "id":0}], "version":1, "error":"Null"}
    else:
      output = {"world":[{"loc": new_loc, "id":S}], "version":1, "error":"Null"}
    # Send output to simulator
    print(output)
    #time.sleep(1) # delays for 5 seconds
    return jsonify(output)
  else:
    return """<html><body>
    GET not implemented
    </body></html>"""

saver = tf.train.Saver()
# Add this to session to see cpu/gpu placement: 
# config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if DoTrain:
  print "Train Source"
  Trainer = Training(sess, correct_prediction[0], logits[0], train_op[0], loss[0],
                     inputs, labels[0], lengths, batch_size, 7) #6
  Trainer.train(training[0], training_labels[0], testing[0], 
                testing_labels[0], generate_batch, training_lens[0],
                testing_lens[0])

  print "Train Reference"
  Trainer = Training(sess, correct_prediction[1], logits[1], train_op[1], loss[1],
                     inputs, labels[1], lengths, batch_size, 5) #2
  Trainer.train(training[1], training_labels[1], testing[1], 
                testing_labels[1], generate_batch, training_lens[1],
                testing_lens[1])

  print "Train Direction"
  Trainer = Training(sess, correct_prediction[2], logits[2], train_op[2], loss[2],
                     inputs, labels[2], lengths, batch_size, 2) #2
  Trainer.train(training[2], training_labels[2], testing[2], 
                testing_labels[2], generate_batch, training_lens[2],
                testing_lens[2])
  
  saver.save(sess, 'model.DARPA-SRD.ckpt')
  print "Saved"
else:
  saver.restore(sess, './model.DARPA-SRD.ckpt')
  print "Loaded Models"
  if __name__ == "__main__" and not DoTrain:
      app.run()
  Vocabulary = {}
  for line in open("JSONReader/data/2016-Version2/SRD/Vocabulary.txt",'r'):
    line = line.strip().split()
    Vocabulary[line[0]] = int(line[1])
