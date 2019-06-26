import gzip
import json
import random
import time
import codecs
from PIL import Image, ImageDraw
from TFLibraries.Ops import *
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import editdistance
random.seed(20161026)
np.set_printoptions(threshold=np.nan)


class Data(object):
  def __init__(self, config, train=True, Training=None, load=True):
    start = time.time()
    self.config = config
    self.batch_size = self.config.batch_size
    self.is_training = Training is None

    # Parameters for image creation
    self.r2_bs = 0.1524
    self.block_size = int(self.config.rep_dim / (2 / self.r2_bs))  # 20 blocks wide
    self.block_size_y = 1 # int(self.config.rep_dim_y / (2 / self.r2_bs))  # 20 blocks wide
    self.half_length = self.block_size / 2  # Half a block
    self.half_length_y = self.block_size_y / 2  # Half a block

    # Build the coordinates of a block so they can be rotated
    self.centered_coords = []  
    for x_offset in range(-1 * self.half_length - 1, self.half_length + 1):
      for z_offset in range(-1 * self.half_length + abs(x_offset) - 1,
                            self.half_length - abs(x_offset) + 1):
        self.centered_coords.append([x_offset, z_offset])
    self.centered_coords = np.array(self.centered_coords).astype(np.float32)

    self.EOS = 1
    self.vocabulary = {"UNK": 0, "EOS": self.EOS}
    self.i_vocabulary = {0: "UNK", self.EOS: "EOS"}

    if load:
      filename = config.training if train else config.evaluation

      if filename is not None:
        if "npz" not in filename:
          self.load_json(filename, Training)
        else:
          data = np.load(filename)
          self.Utterances = data["Utterances"]
          self.Before = data["Before"]
          self.Source = data["Source"]
          self.Target = data["Target"]
          self.SourceID = data["SourceID"]
          loaded_vocab = data["Vocabulary"]
          for w, i in zip(loaded_vocab, range(len(loaded_vocab))):
            self.vocabulary[i] = w
            self.i_vocabulary[w] = i

        """ Create Batches for Training """
        self.indices = []
        self.Utterances, self.Lengths, data_vocab_size, keep, self.mask, heuristic = \
          self._filter_utterances(self.Utterances, trim=False)
        self.Before = self.Before[keep]
        self.Source = self.Source[keep]
        self.SourceID = self.SourceID[keep]
        self.Target = self.Target[keep]
        self.Utterances = self.Utterances[keep]   # Lengths already filtered
        
        self.gold_args = self._choose(heuristic, self.Target, self.FloatWorld[keep])

        self.by_length = []
        for i in range(len(self.Lengths)):
          self.by_length.append((self.Lengths[i], i))
        self.by_length.sort()

        self.size = self.Before.shape[0]

        print "%s loaded %d instances in %4.2f minutes" % (
        filename, self.size, (time.time() - start) / 60)

        """ Build metadata.tsv """
        if Training is None and self.config.load_source_model is None:
          o = codecs.open(self.config.summary_path + "/metadata.tsv", 'w', 'utf-8')
          o.write("Word\tID\n")
          for i in range(len(self.vocabulary)):
            o.write("%s\t%d\n" % (self.i_vocabulary[i], i))
          o.close()


  def toTxt(self, utter):
    return " ".join([self.i_vocabulary[v] for v in utter]).replace("UNK","___")

  def load_json(self, filename, Training=None):
    jsons = [json.loads(line) for line in gzip.open(filename)]

    if Training is None:
      # Build vocabulary
      vocabulary = {}
      for j in jsons:
        for note in j["notes"]:
          if note["type"] == "A0":
            for sent in note["notes"]:
              sent = sent.lower()
              sent = " ".join(sent.split(","))
              sent = " ".join(sent.split("-"))
              sent = " ".join(sent.split("/"))
              for word in TreebankWordTokenizer().tokenize(sent):
                if word not in vocabulary:
                  vocabulary[word] = 0
                vocabulary[word] += 1

      # Threshold & Create Map
      types = [word for word in vocabulary if vocabulary[word] > self.config.oov]
      for i in range(len(self.vocabulary), len(types)):
        self.vocabulary[types[i]] = i
        self.i_vocabulary[i] = types[i]
      self.vocab_size = len(self.vocabulary)
    else:
      self.vocabulary = Training.vocabulary
      self.i_vocabulary = Training.i_vocabulary
      self.vocab_size = Training.vocab_size

    # Create Dense representations
    utterances = []
    before = []
    source = []
    sourceID = []
    target = []
    float_world = []
    for j in jsons:
      has_rotations = "rotations" in j
      worlds = {}
      floats = {}
      rotations = {}
      for i in range(len(j["states"])):
        transformed = np.array(j["states"][i]) + [1,0,1]
        y_axis_rot = self.y_rot(j["rotations"][i]) if has_rotations else None
        worlds[i] = self._draw_world(transformed, y_axis_rot) if not self.config.predict_source else np.zeros((0,0))
        floats[i] = transformed
        rotations[i] = y_axis_rot
      for note in j["notes"]:
        if note["type"] == "A0":
          block_that_moved = self._diff(floats[note["start"]],
                                        floats[note["finish"]])
          start = floats[note["start"]][block_that_moved]
          finish = floats[note["finish"]][block_that_moved]
          if has_rotations:
            start_rot = rotations[note["start"]][block_that_moved]
            finish_rot = rotations[note["finish"]][block_that_moved]
          for sentence in note["notes"]:
            utterances.append(self.convert_text_to_ints(sentence))
            before.append(worlds[note["start"]])
            sourceID.append(block_that_moved)
            float_world.append(floats[note["start"]])

            if has_rotations:
              source.append(np.append(start, start_rot))
              target.append(np.append(finish, finish_rot))
            else:
              source.append(np.append(start, np.zeros((1))))
              target.append(np.append(finish, np.zeros((1))))

    self.size = len(utterances)
    self.Utterances = np.asarray(utterances)
    self.Before = np.asarray(before)
    self.Source = np.asarray(source)
    self.SourceID = np.asarray(sourceID)
    self.Target = np.asarray(target)
    self.FloatWorld = np.asarray(float_world)
    self.Vocabulary = np.asarray([self.i_vocabulary[i] for i in range(len(self.vocabulary))])

    filename = filename.split(".json.gz")[0]
    np.savez(filename, Utterances=self.Utterances, Before=self.Before,
             Source=self.Source, SourceID=self.SourceID, Target=self.Target, 
             Vocabulary=self.Vocabulary, FloatWorld=self.FloatWorld)
    print filename + ".npz saved"

  def convert_text_to_ints(self, sentence):
    """
    Converts words to indices
    """
    sentence = sentence.lower()
    sentence = " ".join(sentence.split(","))
    sentence = " ".join(sentence.split("-"))
    sentence = " ".join(sentence.split("/"))
    return [self.vocabulary[w] if w in self.vocabulary else 0 for w in
            TreebankWordTokenizer().tokenize(sentence)]

  def y_rot(self, quaternions):
    """
      en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    rotations = []
    for x, y, z, w in quaternions:
      ysqr = y*y
      t2 = +2.0 * (w*y - z*x)
      t2 =  1 if t2 > 1 else t2
      t2 = -1 if t2 < -1 else t2
      rotations.append(math.asin(t2))
    return rotations

  def get_n_batches(self, n=1):
    if n == 1:
      return self.gsortet_batch()
    else:
      batches = {}
      for i in range(n):
        dic = self.get_batch()
        for key in dic:
          if key not in batches:
            batches[key] = []
          batches[key].append(dic[key])
    return batches

  def get_batch(self, max_len=None):
    """
    Generate a batch for training with language data
    """
    if len(self.indices) < self.batch_size:
      if max_len is None:
        v = range(self.size)
      else:
        v = [ind for (length, ind) in self.by_length if length < max_len]
      random.shuffle(v)
      self.indices.extend(v)
    r = self.indices[:self.batch_size]
    self.indices = self.indices[self.batch_size:]
    return {"cur_world": self.Before[r],
            "utterances": self.Utterances[r],
            "lengths": self.Lengths[r],
            "target": self.Source[r] if self.config.predict_source else self.Target[r],
            "sourceid": self.SourceID[r],
            "gold_args": self.gold_args[r],
           }

  def get_all_batches(self):
    chunks = [self.batch_size * (i + 1) for i in
              range(self.size / self.batch_size)] # Last batch handled below
    zipped = zip(np.split(self.Before, chunks),
                 np.split(self.Utterances, chunks),
                 np.split(self.Lengths, chunks),
                 np.split(self.Source if self.config.predict_source else self.Target, chunks),
                 np.split(self.SourceID, chunks),
                 np.split(self.gold_args, chunks),
                 np.split(self.FloatWorld, chunks))
    batches = []
    for b, u, l, t, sid, ga, fw in zipped:
      batches.append(
        {"cur_world": b, "utterances": u, "lengths": l, "target": t, 
         "sourceid": sid, "gold_args": ga, "floatworld": fw}
      )

    # Find the leftovers
    leftover = self.size % self.batch_size
    if leftover == 0:
      return batches

    padding = self.batch_size - leftover
    batches[-1]["cur_world"] \
      = Data.const_pad(batches[-1]["cur_world"], ((0, padding), (0,0), (0,0), (0,0)))\
        if not self.config.predict_source else np.zeros((1,1))
    batches[-1]["utterances"] \
       = Data.const_pad(batches[-1]["utterances"], ((0, padding), (0,0)))
    batches[-1]["lengths"] \
      = Data.const_pad(batches[-1]["lengths"], (0, padding))
    batches[-1]["gold_args"] \
      = Data.const_pad(batches[-1]["gold_args"], (0, padding))
    batches[-1]["target"] \
      = Data.const_pad(batches[-1]["target"], ((0, padding), (0,0)))
    batches[-1]["sourceid"] \
      = Data.const_pad(batches[-1]["sourceid"], (0, padding))
    batches[-1]["floatworld"] \
      = Data.const_pad(batches[-1]["floatworld"], (0, padding))
    return batches

  def _filter_utterances(self, utterances, trim=False):
    """
    Need to return:  Utterances, lengths, vocabsize
    """

    # Crop sentences to max allowable length
    max_vocab = 0
    lengths = []
    filtered = np.zeros(shape=[len(utterances), self.config.max_length],
                        dtype=np.int32)
    mask = np.zeros(shape=[len(utterances), 20], dtype=np.float32)
    keep = []
    heuristic = []
    for i in range(len(utterances)):
      utterance = utterances[i]
      if len(utterance) <= self.config.max_length or not self.is_training or trim:
        keep.append(i)
        lengths.append(len(utterance))
        for j in range(min(len(utterance), self.config.max_length)):
          filtered[i][j] = utterance[j]
        options = self._supervised(filtered[i])
        for j in options if len(options) > 0 else range(20):  # All one if no idea
          mask[i][j] = 1
        max_vocab = max(max_vocab, max(utterance))
        heuristic.append(options)
    return filtered, np.array(lengths, dtype=np.int32), max_vocab, np.array(keep), mask, heuristic

  def _choose(self, options, Target, World):
    keep = []
    for ops, target, world in zip(options, Target, World):
      v = [(block_distance(target, world[op]), op) for op in ops if op < len(world)]
      v.sort()
      if len(v) > 0:
        keep.append(v[0][1])
      else:
        keep.append(random.randint(0,19))
    return np.array(keep)

  def _world_to_image(self, world, imgname):
    world_t_flip = np.flipud(world.transpose()) # swap rows with columns to get row-column, then flip up/down so y increases upwards
    world_bin = world_t_flip > 0
    img = Image.fromarray(world_bin.astype(np.uint8)*255).convert('RGB')
    draw = ImageDraw.Draw(img)
    start = np.min(world)
    end = np.max(world)
    for i in range(start,end+1):      
      locs = np.where(world_t_flip == i)
      if len(locs[0]) > 0:
        draw.text((locs[1][0], locs[0][0]), str(i), fill = "green")
    img.save(imgname, 'png')

  def rotate_coords(self, theta, coords):
    # multiply by rotation matrix
    rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return  np.dot(rot, coords.transpose()).transpose()

  def get_coord_row_ranges(self, all_coords):
    min_row = np.min(all_coords[:,0])
    max_row = np.max(all_coords[:,0])
    all_ranges =[]
    for i in range(min_row, max_row+1):
      ys = all_coords[all_coords[:,0] == i,1]
      all_ranges.append([i, np.min(ys), np.max(ys)])
    return all_ranges

  def _draw_world(self, flocation, orientation=None):
    world = np.zeros((self.config.rep_dim_y, self.config.rep_dim, 
                      self.config.rep_dim), dtype=np.int32)
    locations = self.grid(np.array(flocation), [self.config.rep_dim / 2,
                                                self.config.rep_dim_y / 2, 
                                                self.config.rep_dim / 2])
    for block_id in np.random.permutation(len(locations)):
      if orientation is None or orientation[block_id] == 0:
        for x_offset in range(-1 * self.half_length, self.half_length):
          for y_offset in [0]: #range(-1 * self.half_length_y, self.half_length_y):
            for z_offset in range(-1 * self.half_length, self.half_length):
              x_loc = locations[block_id, 0] + x_offset
              y_loc = locations[block_id, 1] + y_offset
              z_loc = locations[block_id, 2] + z_offset
              if 0 <= x_loc < self.config.rep_dim and 0 <= y_loc < self.config.rep_dim_y and 0 <= z_loc < self.config.rep_dim:
                world[y_loc, x_loc, z_loc] = block_id + 1  # Note reordering for conv
      else:
        # apply rotation matrix, then round them back to integers because we're using them for indices
        centered_coords = np.round(self.rotate_coords(orientation[block_id], self.centered_coords)).astype(np.int8)
        # re-organize into [x, y_min, y_max] so we don't end up with holes in our squares
        all_ranges = self.get_coord_row_ranges(centered_coords)
        for i in range(len(all_ranges)):
          x_offset = all_ranges[i][0]
          for z_offset in range(all_ranges[i][1], all_ranges[i][2]+1):
            for y_offset in [0]: #range(-1 * self.half_length_y, self.half_length_y):
              x_loc = locations[block_id, 0] + x_offset
              y_loc = locations[block_id, 1] + y_offset
              z_loc = locations[block_id, 2] + z_offset
              if 0 <= x_loc < self.config.rep_dim and 0 <= y_loc < self.config.rep_dim_y and 0 <= z_loc < self.config.rep_dim:
                world[y_loc, x_loc, z_loc] = block_id + 1  # Note reordering for conv
    return world

  def _diff(self, before, after):
    V = [(block_distance(before[i], after[i]), i) for i in range(len(before))]
    V.sort()
    return V[-1][1]

  def grid(self, locs, dims):
    return np.around(locs * dims).astype(int)  # Range [0,2] --> [0,dim]

  @staticmethod
  def const_pad(mat, pattern):
    return np.pad(mat, pattern, mode='constant', constant_values=0)

  def _supervised(self, utterance):
    sent = self.toTxt(utterance).split()[5:]  # Skip source
    sent.reverse()
    brands = ["adidas", "bmw", "burger king", "coca cola", "esso", "heineken",
              "hp", "mcdonalds", "mercedes benz", "nvidia", "pepsi", "shell",
              "sri", "starbucks", "stella artois", "target", "texaco", "toyota",
              "twitter", "ups"]
    digits = ["one", "two", "three", "four", "five", "six", "seven", "eight",
              "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
              "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
              "twenty"]

    answers = []
    for word in sent:
      for brandid in range(len(brands)):
        brandparts = brands[brandid].split()
        if brands[brandid] == "coca cola":
          brandparts.append("coke")
        for part in brandparts:
          if part == word or (len(word) > 2 > editdistance.eval(part, word)):
            answers.append(brandid)
    if len(answers) > 0:
      return answers

    for word in sent:
      for digitid in range(len(digits)):
        if len(word) > 2 > editdistance.eval(digits[digitid], word):
          answers.append(digitid)
    if len(answers) > 0:
      return answers

    for word in sent:
      for digitid in range(1, len(digits) + 1):
        if str(digitid) == word:
          answers.append(digitid - 1)
    if len(answers) > 0:
      return answers
    return []
