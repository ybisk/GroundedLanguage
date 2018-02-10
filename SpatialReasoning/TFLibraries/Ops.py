import tensorflow as tf
import math

from TFLibraries.Layer import *
from tensorflow.python.framework import ops

tf.set_random_seed(20160905)

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def conv3d(name, l_input, w, b=None, non_linearity='selu', batch_norm=False, 
           training_phase=None, strides=None, trans_shape=None,
           dropout=1.0, padding='SAME', res=None):
  with tf.variable_scope('conv_op') as scope:
    if strides is None:
      strides = [1, 1, 1, 1, 1]
    if trans_shape is not None:
      v = tf.nn.conv3d_transpose(l_input, w, trans_shape, strides=strides,
                                 padding=padding)
    else:
      v = tf.nn.conv3d(l_input, w, strides=strides, padding=padding)
  return conv(name, v, b, non_linearity, batch_norm, training_phase, strides,
              trans_shape, dropout, padding, res)

def conv2d(name, l_input, w, b=None, non_linearity='selu', batch_norm=False, 
           training_phase=None, strides=None, trans_shape=None,
           dropout=1.0, padding='SAME', res=None):
  with tf.variable_scope('conv_op') as scope:
    if strides is None:
      strides = [1, 1, 1, 1]
    if trans_shape is not None:
      v = tf.nn.conv2d_transpose(l_input, w, trans_shape, strides=strides,
                                 padding=padding)
    else:
      v = tf.nn.conv2d(l_input, w, strides=strides, padding=padding)
    
    return conv(name, v, b, non_linearity, batch_norm, training_phase, strides,
                trans_shape, dropout, padding, res)

def conv(name, v, b=None, non_linearity='selu', batch_norm=False, 
         training_phase=None, strides=None, trans_shape=None,
         dropout=1.0, padding='SAME', res=None):
  if b is not None:
    v = tf.nn.bias_add(v, b)
  if batch_norm:
    v = batch_norm_layer(v, training_phase, 'BN' + name)
  if non_linearity == 'relu':
    v = tf.nn.relu(v, name=name)
  elif non_linearity == 'lrelu':
    v = tf.maximum(0.01*v, v, name=name)       # max(av, v)
  elif non_linearity == 'tanh':
    v = tf.nn.tanh(v, name=name)
  elif non_linearity == 'softmax':
    shape = v.get_shape()
    v = tf.reshape(v, [-1, int(shape[3])])      # Flatten
    v = tf.nn.softmax(v, name=name)             # Softmax
    v = tf.reshape(v, shape)                    # Put things back
    return v
  elif non_linearity == 'sigmoid':
    v = tf.nn.sigmoid(v, name=name)
  elif non_linearity == 'selu':
    v = selu(v)

  if res is not None:
    v += res
  v = tf.nn.dropout(v, dropout, seed=20170207)
  return v


def conv2d_trans(name, l_input, shape, w, b, strides=None, padding='SAME'):
  if strides is None:
    strides = [1, 1, 1, 1]
  v = tf.nn.relu(
    tf.nn.bias_add(
      tf.nn.conv2d_transpose(l_input, w, shape, strides=strides,
                             padding=padding), b), name=name)
  return v


def noise(v, std=1e-3):
  return v + tf.random_normal(shape=tf.shape(v), mean=0.0, stddev=std,
                              dtype=tf.float32)


def random_observations(v, minval=0, maxval=21):
  r = tf.to_float(tf.random_uniform(shape=tf.shape(v), minval=minval,
                                    maxval=maxval, dtype=tf.int32))
  b = tf.to_float(tf.equal(tf.random_uniform(shape=tf.shape(v), minval=0,
                                             maxval=20, dtype=tf.int32), 0))
  r = r*b
  mask = 1 - tf.to_float(tf.greater(v, 0))
  r = tf.to_int32(r * mask)
  return v + r


def conv2d_dilated(name, l_input, w, b, rate=2, padding='SAME'):
  v = tf.nn.relu(
    tf.nn.bias_add(
      tf.nn.atrous_conv2d(l_input, w, rate=rate, padding=padding, name=name),
      b), name=name)
  return v


def print_shape(tensor, name, hist=False):
  print "%-30s " % name, tensor.get_shape()
  if hist:
    tf.summary.histogram(name, tensor)


def pearson(x, y):
  mu_x = 1.0 * sum(x) / len(x)
  mu_y = 1.0 * sum(y) / len(y)
  num = sum([(x[i] - mu_x) * (y[i] - mu_y) for i in range(len(x))])
  den = math.sqrt(sum([(v - mu_x) ** 2 for v in x])) * \
        math.sqrt(sum([(v - mu_y) ** 2 for v in y]))
  return num / den


def block_distance(x, y):
  return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(3)])) / 0.1524


# https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py#L110
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, v in grad_and_vars:
      if g is None or v is None:
        print v.name, " has no gradient"
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)
    
    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    
    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def f1(correct, gold, predicted):
  p = 1.0*correct/predicted if predicted > 0 else 0.0
  r = 1.0*correct/gold
  f = 2*p*r/(p+r) if p > 0 or r > 0 else 0.0
  return 100*p, 100*r, 100*f


# http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

