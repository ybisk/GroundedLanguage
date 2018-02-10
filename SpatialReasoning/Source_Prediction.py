import datetime
import sys
import time
import ast

from PIL import Image, ImageDraw
import progressbar

from TFLibraries.Embeddings import Embedding
from TFLibraries.Ops import *

np.set_printoptions(threshold=np.nan)

class Model(object):
  def __init__(self, training, development, synthetic, config):
    self.Training = training
    self.Development = development
    self.config = config
    self.num_objs = 20

    with tf.variable_scope('language'):
      self.text_embeddings = Embedding(self.Training.vocab_size, name='txt',
                                       one_hot=False,
                                       embedding_size=self.config.txt_dim)
    self.dropout = tf.placeholder(tf.float32, name='dropout')
    self.interactive_mode = tf.placeholder(tf.bool, name='interactive')

    # Create language pipeline
    with tf.variable_scope('language') as scope:
      self.placeholders, self.source_loss, self.prediction  = self._model()
      scope.reuse_variables()

    with tf.variable_scope('lang_op'):
      self.language_op = self._language_loss()

    self.opts = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.hist_summary = tf.summary.merge_all()

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Tensorboard
    self.train_summary = tf.summary.scalar('train', self.source_loss)
    self.val_summary = tf.summary.scalar('val', self.source_loss)
    self.writer = tf.summary.FileWriter(self.config.summary_path,
                                        graph=tf.get_default_graph())
    self.saver = tf.train.Saver()

    if config.load_model is not None:
      print "Restoring ", config.load_model
      self.saver.restore(self.sess, config.load_model)
    else:
      print "Initializing"
      self.sess.run(tf.global_variables_initializer())

  def _language_loss(self):
    language_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    return language_optimizer.minimize(self.source_loss)

  def _step(self, step, mode=None):
    feed_dict = {self.dropout: self.config.dropout, 
                 self.interactive_mode: False, 
                }

    batch = self.Training.get_batch()

    for key in self.placeholders:
      feed_dict[self.placeholders[key]] = batch[key]

    tf_ops = [self.source_loss, self.opts]
    tf_ops.append(self.language_op)
    tf_ops.append(self.train_summary)

    vals = self.sess.run(tf_ops, feed_dict=feed_dict)
    loss_val = vals.pop(0)
    opts = vals.pop(0)
    _ = vals.pop(0)

    self.writer.add_summary(vals.pop(0), step)
    if step % 10 == 0:
      #self.writer.add_summary(hist, step)
      self._eval_single_batch(self.Training, step, is_training=True)
    return self.config.batch_size * loss_val, opts

  def interactive(self):
    print "Not Implemented"
    sys.exit()

  def train(self, epochs):
    total_loss = 0.0
    previous_loss = sys.float_info.max
    timestamp = time.time()

    # Progressbar  http://stackoverflow.com/a/3002114
    steps = self.Training.size / self.config.batch_size
    for epoch in range(sum(epochs)):
      bar = progressbar.ProgressBar(maxval=steps + 1,
                                    widgets=[progressbar.Bar('=', '[', ']'),
                                             ' ', progressbar.Percentage()])
      bar.start()

      cur = epoch
      for e, m in zip(self.config.epochs, self.config.modes):
        cur -= e
        if cur < 0:
          mode = m
          break
      for step in range(steps):
        loss, ops = self._step(epoch * steps + step)
        total_loss += loss
        bar.update(step + 1)
        if step % 10 == 0:
          self._eval_single_batch(self.Development, epoch * steps + step,
                                  is_training=False)

      bar.finish()
      current_time = time.time()
      acc = self._eval(self.Development, (epoch + 1) * steps)
      print("E %3d %-8.3f Thres %-5.3f Time %-5.2f -- Dev %5.3f" %
            (epoch, total_loss, (previous_loss - total_loss) / previous_loss,
             (current_time - timestamp) / 60.0, acc))
      timestamp = current_time
      previous_loss = max(total_loss, 1e-10)
      total_loss = 0.0
      self.saver.save(self.sess, self.config.checkpoint_path, epoch)

  def _eval_single_batch(self, eval_data, step, is_training):
    """
    Runs eval on dev/test data with the option to return predictions/performance
    Computes a weighted F1
    """
    feed_dict = {self.dropout: 1.0,
                 self.interactive_mode: False}
    batch = eval_data.get_batch()
    for key in self.placeholders:
      feed_dict[self.placeholders[key]] = batch[key]

    loss, predictions = self.sess.run([self.source_loss, self.prediction], feed_dict=feed_dict)
    gold = batch["sourceid"][0]
    pred = predictions[0]

    if step % 100 == 0 and is_training:
      print "Train\t", self.Training.toTxt(batch["utterances"][0]), "\t\t", loss, gold, pred

    summary = self.sess.run(
                      self.train_summary if is_training else self.val_summary,
                      feed_dict=feed_dict)
    self.writer.add_summary(summary, step)
    return

  def _eval(self, eval_data, step):
    """
    Runs eval on dev/test data with the option to return predictions/performance
    """
    feed_dict = {self.dropout: 1.0,
                 self.interactive_mode: False}
    predicted = []
    gold = []
    utterances = []
    for batch in eval_data.get_all_batches():
      for key in self.placeholders:
        feed_dict[self.placeholders[key]] = batch[key]

      predicted.extend(self.sess.run(self.prediction, feed_dict=feed_dict).tolist())
      utterances.extend(batch["utterances"].tolist())
      gold.extend(batch["sourceid"].tolist())

    v = [(1 if p == g else 0, p, g, u)  for p,g,u in zip(predicted, gold, utterances)]
    acc = 100.0*sum([t for t,p,g,u in v])/len(gold)

    out = open("preds.%d.txt" % step, 'w')
    for a, p, g, u in v:
      out.write("{} {} {}\t{:<50}\n".format(a, p, g, self.Training.toTxt(u)))
    out.close()
    return acc

  def _model(self):
    utterances = tf.placeholder(tf.int32, [self.config.batch_size,
                                           self.config.max_length],
                                name="Utterance")
    lengths = tf.placeholder(tf.int32, [self.config.batch_size], name="Lengths")
    embedded = rnn_fwd(self.config, utterances, lengths, self.text_embeddings,
                       scope="source")
    #embedded, _ = rnn(self.config, utterances, lengths, self.text_embeddings,
    #                  scope="source")
    print_shape(embedded, "embedded")
    w = {
        'w': ff_w(self.config.txt_dim, self.num_objs, 'lang_w_a',
                    reg=self.config.regularizer),
        'b': ff_b(self.num_objs, 'lang_b_a'),
    }
    source = tf.matmul(embedded, w['w']) + w['b']
    print_shape(source, "source")
    prediction = tf.argmax(source, -1)
    answers = tf.placeholder(tf.int32, [self.config.batch_size], name="Answers")
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=source, labels=answers))
    if self.config.regularizer is not None:
      print "Using ", self.config.regularizer
      self.loss += 1e-3 * sum(tf.get_collection(
                              tf.GraphKeys.REGULARIZATION_LOSSES))
    return {"utterances": utterances, "lengths": lengths, "sourceid": answers}, \
            self.loss, prediction


  
