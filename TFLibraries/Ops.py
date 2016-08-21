import tensorflow as tf
import math


def conv2d(name, l_input, w, b, strides=[1, 1, 1, 1], padding='SAME'):
    v = tf.nn.relu(
            tf.nn.bias_add(
              tf.nn.conv2d(l_input, w, strides=strides, padding=padding),
              b), name=name)
    print v
    return v

def conv2d_trans(name, l_input, shape, w, b, strides=[1, 1, 1, 1], padding='SAME'):
    v = tf.nn.relu(
            tf.nn.bias_add(
              tf.nn.conv2d_transpose(l_input, w, shape, strides=strides, padding=padding),
              b), name=name)
    print v
    return v

def pearson(X,Y):
  mu_x = 1.0*sum(X)/len(X)
  mu_y = 1.0*sum(Y)/len(Y)
  num = sum([(X[i] - mu_x)*(Y[i] - mu_y) for i in range(len(X))])
  den = math.sqrt(sum([(x - mu_x)**2 for x in X]))*math.sqrt(sum([(y - mu_y)**2 for y in Y]))
  return num/den

def block_distance(X,Y):
  return math.sqrt(sum([(X[i] - Y[i])**2 for i in range(3)]))/0.1524
