import tensorflow as tf
tf.set_random_seed(20160905)
import numpy as np

class Embedding:
  def __init__(self, vocabsize, one_hot=True, embedding_size=100):
    if one_hot:
      self.embeddings = tf.Variable(np.identity(vocabsize), trainable=False, dtype=tf.float32)
    else:
      self.embeddings = tf.Variable(tf.random_uniform([vocabsize, embedding_size], -1, 1, seed=20160503))

  def lookup(self, X):
    return tf.nn.embedding_lookup(self.embeddings, X, None)
