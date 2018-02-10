import math

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import *

tf.set_random_seed(20160905)

"""
Helper class for quickly creating NN layers with different initializations
"""


def ff_w(input_dim=100, output_dim=100, name='W', init='Xavier', reg=None):
  return variable([input_dim, output_dim], name, init, reg)


def ff_b(dim=100, name='B', init='Zero', reg=None):
  return variable([dim], name, init, reg)


def conv_w(depth=None, width=3, height=3, in_channels=20, out_channels=100,
           name='Conv', init='Xavier', reg=None):
  if depth is None:
    return variable([height, width, in_channels, out_channels],
                    name, init, reg)
  return variable([depth, height, width, in_channels, out_channels],
                  name, init, reg)


def lstm_cell(hidden_dim=100):
  return tf.contrib.rnn.LSTMCell(hidden_dim, state_is_tuple=True,
                                 initializer=tf.contrib.layers.xavier_initializer(
                                   seed=20160501))
def gru_cell(hidden_dim=100):
  return tf.contrib.rnn.GRUCell(hidden_dim)

def gru_cell(hidden_dim=100):
  return tf.contrib.rnn.GRUCell(hidden_dim, state_is_tuple=True,
                                 initializer=tf.contrib.layers.xavier_initializer(
                                   seed=20160501))

# Courtesy of Kevin Shih
def batch_norm_layer(x, train_phase, scope_bn):
  bn = tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True,
                  is_training=train_phase,
                  reuse=None,
                  trainable=True,
                  updates_collections=None,
                  scope=scope_bn)
  return bn     
 
def variable(shape, name, init='Xavier', reg=None):
  regularizer = None
  if reg == 'l2':
    regularizer = tf.contrib.layers.l2_regularizer(1e-2)
  elif reg == 'l1':
    regularizer = tf.contrib.layers.l1_regularizer(1e-3)

  if init == 'Zero':
    vals = np.zeros(shape).astype('f')
    return tf.get_variable(name=name, initializer=vals, dtype=tf.float32)

  if init == 'Xavier':
    w_init = tf.contrib.layers.xavier_initializer(seed=20161016)
  elif init == 'Normal':
    w_init = tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[0]),
                                          seed=12132015)
  elif init == 'Uniform':
    w_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05,
                                           seed=12132015)
  return tf.get_variable(name=name, shape=shape, initializer=w_init,
                         regularizer=regularizer)

def rnn_fwd(config, inputs, lengths, embeddings, scope=''):
    cell = lstm_cell(config.txt_dim)
    cell = DropoutWrapper(cell, output_keep_prob=config.dropout)
    _, encoder = tf.nn.dynamic_rnn(cell, 
                                    inputs=embeddings.lookup(inputs),
                                    sequence_length=lengths, dtype=tf.float32,
                                    scope=scope)
    return encoder[0]


def rnn(config, inputs, lengths, embeddings, condition=None, scope=''):
  """
  Create a sentence representation
  Embed words and run RNN to form sentence representation
  :return: Embedded sentence of size Pixel dim
  """
  if condition is None:
    with tf.variable_scope("enc_lstm%s" % scope):
      fcell = lstm_cell(config.txt_dim)
      bcell = lstm_cell(config.txt_dim)
      fcell = DropoutWrapper(fcell, output_keep_prob=config.dropout)
      bcell = DropoutWrapper(bcell, output_keep_prob=config.dropout)
      _, (encoder_fw_state, encoder_bw_state) \
        = tf.nn.bidirectional_dynamic_rnn(cell_fw=fcell, cell_bw=bcell,
                                          inputs=embeddings.lookup(inputs),
                                          sequence_length=lengths,
                                          dtype=tf.float32)
      fstate = tf.concat([encoder_fw_state[0], encoder_bw_state[0]], 1,
                         name='bidirectional_state')
      return fstate, (encoder_fw_state[0], encoder_bw_state[0])

  else:
    with tf.variable_scope("enc_lstm_condition%s" % scope):
      fcell = lstm_cell(config.txt_dim)
      bcell = lstm_cell(config.txt_dim)
      fcell = DropoutWrapper(fcell, output_keep_prob=config.dropout)
      bcell = DropoutWrapper(bcell, output_keep_prob=config.dropout)
      fstate = LSTMStateTuple(c=tf.zeros((config.batch_size, config.txt_dim)),
                              h=condition[0])
      bstate = LSTMStateTuple(c=tf.zeros((config.batch_size, config.txt_dim)),
                              h=condition[1])
      _, (encoder_fw_state, encoder_bw_state) \
        = tf.nn.bidirectional_dynamic_rnn(cell_fw=fcell, cell_bw=bcell,
                                          initial_state_fw=fstate,
                                          initial_state_bw=bstate,
                                          inputs=embeddings.lookup(inputs),
                                          sequence_length=lengths,
                                          dtype=tf.float32)
      fstate = tf.concat([encoder_fw_state[0], encoder_bw_state[0]], 1,
                         name='bidirectional_state')
      return fstate, (encoder_fw_state[0], encoder_bw_state[0])

