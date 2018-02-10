import tensorflow as tf
import numpy as np

from TFLibraries.Ops import *

tf.set_random_seed(20160905)
np.random.seed(20160905)


class Embedding:
  def __init__(self, vocabsize, name='embed', one_hot=True, zero_unk=False,
               embedding_size=100, orthogonal=False):
    if orthogonal:
      self.embeddings = tf.get_variable(name=name, initializer=tf.orthogonal_initializer(seed=20170607), shape=[vocabsize, embedding_size])
    else:
      if one_hot:
        vals = np.identity(vocabsize).astype('f')
        if zero_unk:
          vals[0] = 0
        self.embeddings = tf.get_variable(name, initializer=vals, trainable=False,
                                          dtype=tf.float32)
      else:
        vals = np.random.uniform(-0.05, 0.05,
                                 size=(vocabsize, embedding_size)).astype('f')
        if zero_unk:
          vals[0] = 0
        self.embeddings = tf.get_variable(name, initializer=vals,
                                          dtype=tf.float32)

    print_shape(self.embeddings, "%s embedding matrix" % name)

  def lookup(self, x, name="embed_lookup"):
    return tf.nn.embedding_lookup(self.embeddings, x, name=name)
