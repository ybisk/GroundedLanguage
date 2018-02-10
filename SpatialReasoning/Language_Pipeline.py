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
    self.num_objs = 20    # 20 blocks + Background

    with tf.variable_scope('language'):
      self.text_embeddings = Embedding(self.Training.vocab_size, name='txt',
                                       one_hot=False,
                                       embedding_size=self.config.txt_dim)
    with tf.variable_scope('ops'):
      self.op_embeddings = Embedding(self.config.num_ops, name='ops',
                                     one_hot=False,
                                     embedding_size=self.config.pixel_dim)

    self.locs = []
    self.vision_bn_phase = tf.placeholder(tf.bool, name='train_phase')
    self.dropout = tf.placeholder(tf.float32, name='dropout')
    self.interactive_mode = tf.placeholder(tf.bool, name='interactive')

    # Create language pipeline
    with tf.variable_scope('language') as scope:
      l_phs, self.vision_loss, (pred_x, pred_y, pred_z, pred_t) = self._model()
      scope.reuse_variables()
    self.locs.extend([pred_x, pred_y, pred_z, pred_t])

    self.placeholders = l_phs

    with tf.variable_scope('lang_op'):
      self.language_op = self._language_loss()

    self.opts = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.hist_summary = tf.summary.merge_all()

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Tensorboard
    self.train_reg_summary = tf.summary.scalar('train_reg', self.reg_loss)
    self.train_rot_summary = tf.summary.scalar('train_rot', self.rotation_loss)
    self.val_cost_summary = tf.summary.scalar('val_cost', self.reg_loss)
    self.val_rot_summary = tf.summary.scalar('val_rot', self.rotation_loss)
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
    self.language_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    language_op = self.language_optimizer.minimize(self.vision_loss)
    return language_op

  def _step(self, step, mode=None):
    feed_dict = {self.dropout: self.config.dropout, 
                 self.interactive_mode: False, 
                 self.interactive_arg: np.zeros((self.config.batch_size,20)), 
                 self.interactive_op: np.zeros((self.config.batch_size,32))}

    batch = self.Training.get_batch()
    feed_dict[self.vision_bn_phase] = True

    for key in self.placeholders:
      feed_dict[self.placeholders[key]] = batch[key]

    tf_ops = [self.loss, self.opts, self.hist_summary]
    tf_ops.append(self.language_op)
    tf_ops.append(self.train_reg_summary)
    tf_ops.append(self.train_rot_summary)

    vals = self.sess.run(tf_ops, feed_dict=feed_dict)
    loss_val = vals.pop(0)
    opts = vals.pop(0)
    hist = vals.pop(0)
    _ = vals.pop(0)

    self.writer.add_summary(vals.pop(0), step)
    self.writer.add_summary(vals.pop(0), step)
    if step % 10 == 0:
      self.writer.add_summary(hist, step)
      self._eval_single_batch(self.Training, step, is_training=True)
    return self.config.batch_size * loss_val, opts

  def interactive(self):
    self.config.batch_size = 1
    to_run = self.locs
    to_run.append(self.img_op)
    to_run.append(self.img_pred)
    while True:
      world = ast.literal_eval(raw_input("world: "))
      rotations = ast.literal_eval(raw_input("rotations: "))
      arg_dist = ast.literal_eval(raw_input("arg dist: "))
      op_dist = ast.literal_eval(raw_input("op dist: "))

      feed_dict = {}
      feed_dict = {self.vision_bn_phase: False, 
                   self.dropout: 1.0, 
                   self.interactive_mode: True,
                   self.placeholders["utterances"]: np.zeros(shape=[1,self.config.max_length], dtype=np.int32),
                   self.placeholders["lengths"]: np.zeros(shape=(1,), dtype=np.int32),
                   self.placeholders["target"]: np.zeros(shape=[1,4], dtype=np.float32),
                   self.placeholders["cur_world"]: self.Training._draw_world(world, rotations).reshape(1, self.config.rep_dim_y, self.config.rep_dim, self.config.rep_dim),
                   self.interactive_arg: np.array(arg_dist).reshape((1,20)),
                   self.interactive_op: np.array(op_dist).reshape((1,32))
                  }

      x,y,z,t, img_ops, img_preds = self.sess.run(to_run, feed_dict=feed_dict)
      print x,y,z,t
      img = Image.fromarray((np.squeeze(img_ops)* 255).astype(np.uint8)).convert('RGB')
      draw = ImageDraw.Draw(img)
      img.save('ops.png', 'png')

      img = Image.fromarray((np.squeeze(img_preds)*255).astype(np.uint8)).convert('RGB')
      draw = ImageDraw.Draw(img)
      img.save('preds.png', 'png')

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
      mean, median, rotation = self._eval(self.Development, (epoch + 1) * steps)
      print("E %3d %-8.3f Thres %-5.3f Time %-5.2f -- Dev %5.3f %5.3f %5.3f" %
            (epoch, total_loss, (previous_loss - total_loss) / previous_loss,
             (current_time - timestamp) / 60.0, mean, median, rotation))
      timestamp = current_time
      previous_loss = max(total_loss, 1e-10)
      total_loss = 0.0
      self.saver.save(self.sess, self.config.checkpoint_path, epoch)

  def _eval_single_batch(self, eval_data, step, is_training):
    """
    Runs eval on dev/test data with the option to return predictions/performance
    Computes a weighted F1
    """
    feed_dict = {self.vision_bn_phase: False, 
                 self.dropout: 1.0, 
                 self.interactive_mode: False, 
                 self.interactive_arg: np.zeros((self.config.batch_size,20)), 
                 self.interactive_op: np.zeros((self.config.batch_size,32))}
    batch = eval_data.get_batch()
    for key in self.placeholders:
      feed_dict[self.placeholders[key]] = batch[key]

    loss = self.sess.run(self.vision_loss, feed_dict=feed_dict)

    if step % 100 == 0 and is_training:
      print "Train\t", self.Training.toTxt(batch["utterances"][0]), "\t\t", loss

    summary = self.sess.run(
        [self.train_reg_summary if is_training else self.val_cost_summary,
        self.train_rot_summary if is_training else self.val_rot_summary],
        feed_dict=feed_dict)
    for smy in summary:
      self.writer.add_summary(smy, step)
    return

  def _eval(self, eval_data, step):
    """
    Runs eval on dev/test data with the option to return predictions/performance
    """
    feed_dict = {self.vision_bn_phase: False, 
                 self.dropout: 1.0, 
                 self.interactive_mode: False, 
                 self.interactive_arg: np.zeros((self.config.batch_size,20)), 
                 self.interactive_op: np.zeros((self.config.batch_size,32))}
    predicted = []
    gold = []
    utterances = []
    for batch in eval_data.get_all_batches():
      for key in self.placeholders:
        feed_dict[self.placeholders[key]] = batch[key]

      x, y, z, t = self.sess.run(self.locs, feed_dict=feed_dict)
      utterances.extend(batch["utterances"].tolist())
      for a, b, c, d in zip(x, y, z, t):
        predicted.append((a, b, c, d))
      gold.extend(batch["target"].tolist())

    dists = [block_distance(a, b) for a, b in \
             zip(predicted[:eval_data.size], gold[:eval_data.size])]
    angles = [abs(a[3] - b[3]) for a, b in \
             zip(predicted[:eval_data.size], gold[:eval_data.size])]

    v = [(d, a, u, g, p) for d, a, u, g, p in
         zip(dists, angles, utterances, gold, predicted)]
    v.sort()

    out = open("preds.%d.txt" % step, 'w')
    for d, a, u, g, p in v:
      out.write("{} {}\t{} {}\t{:<50}\n".format(d, a, g, p, self.Training.toTxt(u)))
    out.close()
    dists.sort()
    return sum(dists) / len(dists), dists[len(dists) / 2], sum(angles)/len(angles)

  def _grid(self, x, z):
    scaled_x = x * self.config.rep_dim / 2
    scaled_z = z * self.config.rep_dim / 2
    return self.config.rep_dim * tf.cast(scaled_x, tf.int32) + \
           tf.cast(scaled_z, tf.int32)

  def _kl_match(self, world, value):
    """
    Every dimension gets an activation equal to predicted probability
    """
    one_hot_world = tf.one_hot(world, 21)
    print_shape(one_hot_world, "one hot world")
    print_shape(value, "predicted block")
    value = tf.concat((tf.zeros((self.config.batch_size,1)), value), -1)
    value = tf.reshape(value, [self.config.batch_size, 1, 1, 1, -1])
    # ideally just use cross entropy here
    attention = tf.reduce_sum(10 * one_hot_world * value, -1)
    print_shape(attention, "attention")

    mask = tf.cast(tf.greater(world, 0), tf.float32)
    attention *= mask
    return tf.reshape(attention, [-1, self.config.rep_dim_y, 
                                  self.config.rep_dim, 
                                  self.config.rep_dim, 
                                  1])
  def _language(self):
    """
    Encodes (and decodes) a sentence.  These are then transformed into
    distributions for argument and operation prediction.
    :input  sentences and their lengths
    :return distributiosn
    """
    utterances = tf.placeholder(tf.int32, [self.config.batch_size,
                                           self.config.max_length],
                                name="Utterance")
    lengths = tf.placeholder(tf.int32, [self.config.batch_size], name="Lengths")
    w = {
        'w_a': ff_w(2 * self.config.txt_dim, self.num_objs, 'lang_w_a',
                    reg=self.config.regularizer),
        'w_o': ff_w(2 * self.config.txt_dim, self.config.num_ops, 'lang_w_o',
                    reg=self.config.regularizer),
        'b_a': ff_b(self.num_objs, 'lang_b_a'),
        'b_o': ff_b(self.config.num_ops, 'lang_b_o'),
    }
    embedded, _ = rnn(self.config, utterances, lengths, self.text_embeddings,
                      scope="args")
    argument_dist = tf.nn.softmax(tf.matmul(embedded, w['w_a']) + w['b_a'])
    embedded_o, _ = rnn(self.config, utterances, lengths, self.text_embeddings,
                        scope="op")
    operation_dist = tf.nn.softmax(tf.matmul(embedded_o, w['w_o']) + w['b_o'])

    #argument_dist = tf.Print(argument_dist, [argument_dist, operation_dist], summarize=100000)
    print_shape(argument_dist, "argument", True)
    print_shape(operation_dist, "operation", True)
    return [utterances, lengths], [argument_dist, operation_dist]

  def _apply_operation(self, dist_o, world):
    """
    Choose an operation from embedding matrix and apply it to the world.
    We then run two convolutions to help summarize regions
    """
    # Batch, X, Y, Z, F
    before = tf.reduce_mean(tf.reduce_mean(world, -1, keep_dims=True) / 
                            tf.reduce_max(world), 1) # Collapse Y
    self.img_op = tf.concat((before, before, before), -1)
    print_shape(world, "world input")
    world = tf.reshape(world, [self.config.batch_size, -1, 1])
    print_shape(world, "world reshape")
    operation = tf.matmul(dist_o, self.op_embeddings.embeddings)
    world = tf.matmul(world, tf.reshape(operation,
                                        [self.config.batch_size, 1, self.config.pixel_dim]))

    print_shape(world, "world postop")
    world = tf.reshape(world, [self.config.batch_size, self.config.rep_dim_y, 
                               self.config.rep_dim, self.config.rep_dim, -1])
    print_shape(world, "world reshape")

    w = {
        'c1': conv_w(self.config.kernel_size_y, 
                     self.config.kernel_size, 
                     self.config.kernel_size, 
                     self.config.pixel_dim, 
                     self.config.hidden_dim, 
                     'filt', reg=self.config.regularizer),
        'c2': conv_w(max(self.config.kernel_size_y - 2, 1), 
                     max(self.config.kernel_size - 2, 1), 
                     max(self.config.kernel_size - 2, 1), 
                     self.config.hidden_dim, 
                     self.config.hidden_dim, 
                     'filt2', reg=self.config.regularizer)

    }
    world = conv3d('conv', world, w['c1'],
                   non_linearity=self.config.non_linearity,
                   batch_norm=self.config.batch_norm,
                   training_phase=self.vision_bn_phase)
    print_shape(world, "world c1")
    world = conv3d('conv2', world, w['c2'], b=None,
                   non_linearity=self.config.non_linearity,
                   batch_norm=self.config.batch_norm,
                   training_phase=self.vision_bn_phase)
    print_shape(world, "world c2")

    after = tf.reduce_mean(tf.reduce_mean(world, -1, keep_dims=True) / 
                           tf.reduce_max(world), 1) # Collapse Y
    tf.summary.image("att_op", self.img_op)
    print_shape(world, "post op")
    return world

  def _xyzt(self, world):
    """
    Run a conv @ every pixel to predict an offset and confidence
    """
    # pixel-wise prediction conv layer
    pred = conv_w(1, 1, 1, self.config.hidden_dim, 8, 'pred', reg=self.config.regularizer)

    pred_b = ff_b(8, 'embed3_b')
    conf = conv3d('conf', world, pred, b=pred_b, non_linearity='none',
                  batch_norm=False, training_phase=self.vision_bn_phase,
                  strides=[1, 1, 1, 1, 1], padding='SAME')
    print_shape(conf, "local pred")
    flat_conf = tf.reshape(conf, [self.config.batch_size, -1, 8])

    # FIXME - Are these dimensions correct x,y,z vs y,x,z ?
    cx = tf.nn.softmax(tf.squeeze(tf.slice(flat_conf, [0, 0, 0], [-1, -1, 1])))
    dx = tf.squeeze(tf.slice(flat_conf, [0, 0, 1], [-1, -1, 1]))
    cy = tf.nn.softmax(tf.squeeze(tf.slice(flat_conf, [0, 0, 2], [-1, -1, 1])))
    dy = tf.squeeze(tf.slice(flat_conf, [0, 0, 3], [-1, -1, 1]))
    cz = tf.nn.softmax(tf.squeeze(tf.slice(flat_conf, [0, 0, 4], [-1, -1, 1])))
    dz = tf.squeeze(tf.slice(flat_conf, [0, 0, 5], [-1, -1, 1]))
    ct = tf.nn.softmax(tf.squeeze(tf.slice(flat_conf, [0, 0, 6], [-1, -1, 1])))
    dt = tf.tanh(tf.squeeze(tf.slice(flat_conf, [0, 0, 7], [-1, -1, 1])))*3.141592653589793

    step = 2.0 / self.config.rep_dim
    span = np.arange(0, 2, step, dtype=np.float32)
    span_y = np.arange(0, 1, 1.0 / self.config.rep_dim_y, dtype=np.float32)
    y,x,z = tf.meshgrid(span_y, span, span, indexing='ij')
    x = tf.reshape(x, [1, -1])
    y = tf.reshape(y, [1, -1])
    z = tf.reshape(z, [1, -1])
    x_pred = tf.reduce_sum(cx * (x + dx), axis=-1)
    y_pred = tf.reduce_sum(cy * (y + dy), axis=-1)
    z_pred = tf.reduce_sum(cz * (z + dz), axis=-1)
    t_pred = tf.reduce_sum(ct * dt, axis=-1)

    # Visualize
    pred_loc = self._grid(x_pred, z_pred)
    pred_loc = tf.reshape(tf.one_hot(pred_loc, self.config.rep_dim ** 2 * 
                                               self.config.rep_dim_y),
                          [self.config.batch_size, 
                           self.config.rep_dim_y, 
                           self.config.rep_dim,
                           self.config.rep_dim, 1])
    pred_loc = tf.reduce_max(pred_loc, 1)
    soft_conf = cx + cz
    att_loc = tf.reshape(soft_conf / tf.reduce_max(soft_conf),
                         [self.config.batch_size, 
                          self.config.rep_dim_y,
                          self.config.rep_dim, 
                          self.config.rep_dim, 1])
    att_loc = tf.reduce_max(att_loc, 1)
    return (x_pred, y_pred, z_pred, t_pred), (att_loc, pred_loc)

  def _model(self):
    # Get a prediction for arguments and operations to attend to from language
    [utterances, lengths], [distribution_a, distribution_o] = self._language()

    # Interactive mode
    self.interactive_arg = tf.placeholder(tf.float32, [self.config.batch_size, self.num_objs], name='iarg')
    self.interactive_op = tf.placeholder(tf.float32, [self.config.batch_size, self.config.num_ops], name='iop')

    dist_a = tf.cond(self.interactive_mode, lambda: self.interactive_arg, lambda: distribution_a)
    dist_o = tf.cond(self.interactive_mode, lambda: self.interactive_op, lambda: distribution_o)

    # Attend to the world based on language predictions
    cur_world = tf.placeholder(tf.int32,
                               [self.config.batch_size, self.config.rep_dim_y,
                                self.config.rep_dim, self.config.rep_dim], 
                               name="CurWorld")
    atts = self._kl_match(cur_world, dist_a)

    # Apply chosen operation
    world = self._apply_operation(dist_o, atts)

    # XYZ Prediction
    (x, y, z, t), (att_loc, pred_loc) = self._xyzt(world)

    # Regression loss
    answers = tf.placeholder(tf.float32, [self.config.batch_size, 4],
                             name="answers")
    reg_loss_full = tf.square((x - answers[:, 0]) / 0.1524) + \
                    tf.square((y - answers[:, 1]) / 0.1524) + \
                    tf.square((z - answers[:, 2]) / 0.1524)
    self.reg_loss = tf.reduce_mean(tf.sqrt(reg_loss_full))

    self.rotation_loss = tf.reduce_mean(tf.atan(tf.sin(t - answers[:, 3])/tf.cos(t - answers[:, 3])))   #FIXME, should be atan2(sin, cos)
    entropy = 1e-3 * tf.reduce_mean(-1 * dist_a * tf.log(dist_a)) #+ tf.reduce_mean(-1 * dist_o * tf.log(dist_o)))
    count = 1e-2 * tf.reduce_mean(tf.nn.relu(tf.reduce_sum(tf.round(dist_a), -1) - 2))
    self.loss = self.reg_loss + entropy # self.rotation_loss

    if self.config.regularizer is not None:
      print "Using ", self.config.regularizer
      self.loss += 1e-3 * sum(tf.get_collection(
                              tf.GraphKeys.REGULARIZATION_LOSSES))

    # Visualization (2D)
    gold_loc = self._grid(answers[:, 0], answers[:, 2])
    gold_loc = tf.reshape(tf.one_hot(gold_loc, self.config.rep_dim ** 2),
                          [self.config.batch_size, self.config.rep_dim,
                           self.config.rep_dim, 1])
    gold_loc += att_loc
    pred_loc += att_loc
    pred_world = tf.concat((pred_loc, gold_loc, att_loc), axis=-1)
    self.img_pred = pred_world
    tf.summary.image("pred", pred_world)

    # Placeholders, Loss, Predictions
    return {"cur_world": cur_world, "utterances": utterances,
            "lengths": lengths, "target": answers}, self.loss, [x, y, z, t]
