import tensorflow as tf
tf.set_random_seed(20160905)
import math


class Layers:
  """
  Helper class for quickly creating NN layers with differen initializations
  """

  def W(self, input_dim=100, output_dim=100, name='W', init='Xavier'):
    if init == 'Normal':
      return self.normal([input_dim, output_dim], name)
    elif init == 'Uniform':
      return self.uniform([input_dim, output_dim], name)
    else:
      return self.xavier([input_dim, output_dim], name)

  def b(self, dim=100, name='B', init='Xavier'):
    if init == 'Normal':
      return self.normal([dim], name)
    if init == 'Uniform':
      return self.uniform([dim], name)
    if init == 'Xavier':
      return self.xavier([dim], name)
    if init == 'Zero':
      return self.zero([dim], name)

  def convW(self, shape, name='Conv', init='Xavier'):
    """
    Convolution weights
    """
    if init == 'Xavier':
      return self.xavier(shape, name)
    if init == 'Normal':
      return self.normal(shape, name)
    if init == 'Uniform':
      return self.uniform(shape, name)

  def normal(self, shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=1.0 / math.sqrt(shape[0]), seed=12132015), name=name)
  def xavier(self, shape, name):
    init_range = math.sqrt(6.0 / sum(shape))
    return tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range, seed=20160429), name)
  def uniform(self, shape, name):
    return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05, seed=12132015), name=name)
  def zero(self, shape, name):
    return tf.zeros(shape, name=name)
