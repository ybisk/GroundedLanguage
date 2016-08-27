import numpy as np
from PIL import Image
from TFLibraries.Ops import *


class Eval:

  def __init__(self, session, rep_dim, unit_size, space_size, batch_size,
               current_world, next_world, utterance, utterance_length,
               logits, correct_prediction):
    self.cur_world = current_world
    self.next_world = next_world
    self.inputs = utterance
    self.lengths = utterance_length
    self.sess = session

    self.batch_size = batch_size
    self.rep_dim = rep_dim
    self.unit_size = unit_size
    self.space_size = space_size

    self.logits = logits
    self.correct_prediction = correct_prediction

  """ Evaluation Functions """
  def SMeval(self, DWi, DU, Dlens, DWj, keep_predictions=False):
    """
    Runs eval on dev/test data with the option to return predictions or performance
    """
    predictions = []
    for i in range(len(DWi)/self.batch_size):
      batch_range = range(self.batch_size*i,self.batch_size*(i+1))
      wi = DWi[batch_range]
      wj = DWj[batch_range]
      U = DU[batch_range]
      lens = Dlens[batch_range]
      feed_dict = {self.cur_world: wi, self.next_world: wj, self.inputs: U, self.lengths: lens}
      if keep_predictions:
        predictions.extend(self.sess.run(tf.argmax(self.logits,1), feed_dict))
      else:
        predictions.extend(self.sess.run(self.correct_prediction, feed_dict))

    ## Grab the extras
    last_chunk = self.batch_size*(i+1)
    gap = self.batch_size - (len(DWi) - last_chunk)
    wi = np.pad(DWi[last_chunk:], ((0,gap),(0,0), (0,0), (0,0)), mode='constant', constant_values=0)
    wj = np.pad(DWj[last_chunk:], ((0,gap),(0,0)), mode='constant', constant_values=0)
    U = np.pad(DU[last_chunk:], ((0,gap),(0,0)), mode='constant', constant_values=0)
    lens = np.pad(Dlens[last_chunk:], ((0,gap)), mode='constant', constant_values=0)
    feed_dict = {self.cur_world: wi, self.next_world: wj, self.inputs: U, self.lengths: lens}
    if keep_predictions:
      predictions.extend(self.sess.run(tf.argmax(self.logits,1), feed_dict)[:self.batch_size - gap])
      return predictions
    else:
      predictions.extend(self.sess.run(self.correct_prediction, feed_dict)[:self.batch_size - gap])
      return 100.0*sum(predictions)/len(predictions)


  def convertToReals(self, x,y):
    x *= self.unit_size
    y *= self.unit_size
    x -= self.space_size/2
    y -= self.space_size/2
    y *= -1
    return (x,y)

  def real_eval(self, DWi, DU, Dlens, DWj, real_dev, real_dev_id):
    # These are 1-hot representations
    predictions = self.SMeval(DWi, DU, Dlens, DWj, keep_predictions=True)
    # convert predictions to (idx,idy)
    idx_idy_pairs = [(p/self.rep_dim, p%self.rep_dim) for p in predictions]
    gold_idx_idy_pairs = [(p/self.rep_dim, p%self.rep_dim) for p in np.argmax(DWj, axis=1)] ## This returns the first (lower corner? not center?)
    # convert to real values
    xy_pairs = [self.convertToReals(x,y) for (x,y) in idx_idy_pairs]
    gold_xy_pairs = [self.convertToReals(x,y) for (x,y) in gold_idx_idy_pairs]
    # Evaluate
    Gold_locs = []
    P = []
    for i in range(len(real_dev_id)):
      P.append(block_distance([xy_pairs[i][0], 0.1, xy_pairs[i][1]],
                              real_dev[i][3*real_dev_id[i]:3*(real_dev_id[i]+1)]))
      Gold_locs.append(real_dev[i][3*real_dev_id[i]:3*(real_dev_id[i]+1)])
    P.sort()
    G = []
    for i in range(len(real_dev_id)):
      G.append(block_distance([gold_xy_pairs[i][0], 0.1, gold_xy_pairs[i][1]],
                              real_dev[i][3*real_dev_id[i]:3*(real_dev_id[i]+1)]))
    G.sort()
    #print gold_xy_pairs
    #print Gold_locs
    return (sum(P)/len(P), P[len(P)/2], sum(G)/len(G), G[len(G)/2])

  def createImage(self, name, before, after, matrix, dim):
    """
      Creates an image of the confidences
    """
    img = Image.new('RGB', (dim,dim), "black")
    pixels = img.load()
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        pixels[i,j] = (int(255*before[i][j]),
                       int(255*matrix[i][j]),
                       int(255*after[i][j]))
    img.save(name)
    #img.show()

