import numpy as np
import tensorflow as tf
class Training:

  def __init__(self, sess, correct_prediction, logits, optimizer, loss, dataset, 
               labels, lengths=None, batch_size=128, epochs=10):
    self.sess = sess
    self.correct_prediction = correct_prediction
    self.logits = logits
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.loss = loss
    self.dataset = dataset
    self.labels = labels
    self.lengths = lengths
    self.epochs = epochs

  def eval(self, data, label, lens):
    predictions = []
    vals = []
    for i in range(data.shape[0]/self.batch_size):
      D = data[range(self.batch_size*i,self.batch_size*(i+1))]
      L = label[range(self.batch_size*i,self.batch_size*(i+1))]
      if lens is not None:
        l = lens[range(self.batch_size*i,self.batch_size*(i+1))]
        feed_dict={self.dataset:D, self.labels:L, self.lengths:l}
      else:
        feed_dict={self.dataset:D, self.labels:L}
      predictions.extend(self.sess.run(self.correct_prediction, feed_dict))
      vals.extend(self.sess.run(tf.argmax(self.logits,1), feed_dict))

    ## DO THE EXTRA
    last_chunk = self.batch_size*(i+1)
    gap = self.batch_size - (data.shape[0] - last_chunk)
    D = np.pad(data[last_chunk:], ((0,gap),(0,0)), mode='constant', constant_values=0)
    L = np.pad(label[last_chunk:], ((0,gap),(0,0)), mode='constant', constant_values=0)
    if lens is not None:
      l = np.pad(lens[last_chunk:], (0,gap), mode='constant', constant_values=0)
      feed_dict={self.dataset:D, self.labels:L, self.lengths:l}
    else:
      feed_dict={self.dataset:D, self.labels:L}
    predictions.extend(self.sess.run(self.correct_prediction, feed_dict)[:self.batch_size - gap])
    vals.extend(self.sess.run(tf.argmax(self.logits,1), feed_dict)[:self.batch_size - gap])

    print vals

    ## PRINT THE PREDICTONS
    return 100.0*sum(predictions)/len(predictions)

  def train(self, train, train_labels, dev, dev_labels, generate_batch, train_lens=None, dev_lens=None):
    num_epochs = 10
    print('Initialized')
    total_loss = 0.0
    for epoch in range(self.epochs):
      for step in range(train.shape[0]/self.batch_size):
        if train_lens is None:
          batch_data, batch_labels = generate_batch(self.batch_size, train, train_labels)
          feed_dict = {self.dataset: batch_data, self.labels: batch_labels}
        else:
          batch_data, batch_labels, batch_lens = generate_batch(self.batch_size, train, train_labels, train_lens)
          feed_dict = {self.dataset: batch_data, self.labels: batch_labels, self.lengths:batch_lens}
        _, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        total_loss += l
      #print('Epoch %d: %f  %f  %f' % (epoch, total_loss, self.eval(train, train_labels, train_lens), self.eval(dev, dev_labels, dev_lens)))
      print('Epoch %d: %f  %f' % (epoch, total_loss, self.eval(dev, dev_labels, dev_lens)))
      total_loss = 0.0
