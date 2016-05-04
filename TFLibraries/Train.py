import numpy as np
class Training:

  def __init__(self, sess, correct_prediction, optimizer, loss, dataset, 
               labels, lengths, batch_size):
    self.sess = sess
    self.correct_prediction = correct_prediction
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.loss = loss
    self.dataset = dataset
    self.labels = labels
    self.lengths = lengths

  def eval(self, data, label, lens):
    predictions = []
    for i in range(data.shape[0]/self.batch_size):
      D = data[range(self.batch_size*i,self.batch_size*(i+1))]
      L = label[range(self.batch_size*i,self.batch_size*(i+1))]
      l = lens[range(self.batch_size*i,self.batch_size*(i+1))]
      predictions.extend(self.sess.run(self.correct_prediction, 
                          feed_dict={self.dataset:D, self.labels:L, self.lengths:l}))
    return 100.0*sum(predictions)/len(predictions)

  def train(self, train, train_labels, dev, dev_labels, generate_batch, train_lens=None, dev_lens=None):
    num_epochs = 25
    print('Initialized')
    total_loss = 0.0
    for epoch in range(num_epochs):
      for step in range(train.shape[0]/self.batch_size):
        batch_data, batch_labels, batch_lens = generate_batch(self.batch_size, train, train_labels, train_lens)
        feed_dict = {self.dataset: batch_data, self.labels: batch_labels, self.lengths:batch_lens}
        _, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        total_loss += l
      print('Epoch %d: %f  %f  %f' % (epoch, total_loss, self.eval(train, train_labels, train_lens), self.eval(dev, dev_labels, dev_lens)))
      total_loss = 0.0
