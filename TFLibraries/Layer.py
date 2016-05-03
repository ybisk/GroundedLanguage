import tensorflow as tf
import math


class Layers:
  """
  Helper class for quickly creating NN layers with differen initializations
  """

  def W(self, input_dim=100, output_dim=100, name='W', init='Xavier'):
    if init == 'Normal':
      return self.normal_W(input_dim, output_dim, name)
    elif init == 'Uniform':
      return self.uniform_W(input_dim, output_dim, name)
    else:
      return self.xavier_W(input_dim, output_dim, name)

  def b(self, dim=100, name='B', init='Xavier'):
    if init == 'Normal':
      return self.normal_b(dim, name)
    elif init == 'Uniform':
      return self.uniform_b(dim, name)
    else:
      return self.xavier_b(dim, name)


  def normal_W(self, input_dim=100, output_dim=100, name='W'):
    return tf.Variable(tf.random_normal([input_dim, output_dim], stddev=1.0 / math.sqrt(input_dim), seed=12132015), name=name)
  def normal_b(self, dim=100, name='B'):
    return tf.Variable(tf.random_normal([dim], stddev=1.0 / math.sqrt(dim), seed=12132015), name=name)

  def xavier_W(self, input_dim=100, output_dim=100, name='W'):
    init_range = math.sqrt(6.0 / (input_dim + output_dim))
    return tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, seed=20160429), name)
  def xavier_b(self, dim=100, name='B'):
    init_range = math.sqrt(6.0 / dim)
    return tf.Variable(tf.random_uniform([dim], minval=-init_range, maxval=init_range, seed=20160429), name)

  def uniform_W(self, input_dim=100, output_dim=100, name="W"):
    return tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=12132015), name=name)
  def uniform_b(self, dim=100, name="B"):
    return tf.Variable(tf.random_uniform([dim], minval=-0.05, maxval=0.05, seed=12132015), name=name)
