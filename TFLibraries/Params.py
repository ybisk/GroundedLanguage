class Param:

  batch_size = 512
  maxlength = 40
  filters = 4
  multicells = 1
  hiddendim = 100
  num_epochs = 12
  dropout = 0.75

  rep_dim = 32
  offset = rep_dim/2 -1
  block_size = 0.1528
  space_size = 3.0
  unit_size = space_size / rep_dim

  Directory = '/home/ybisk/GroundedLanguage'
  TrainData = 'Priors/Train.%d.L1.LangAndBlank.20.npz' % rep_dim
  EvalData = 'Priors/Dev.%d.L1.Lang.20.npz' % rep_dim
  RawEval = 'Priors/WithText/Dev.mat.gz'
  VocabFile = 'JSONReader/data/2016-NAACL/SRD/Vocabulary.txt'
  #EvalData = 'Priors/Test.Lang.20.npz'
  #RawEval = 'Priors/WithText/Test.mat.gz'

  
  def __init__(self, filename):
    for line in open(filename):
      if len(line) < 2 or line[0] == "#":
        continue
      line = line.strip().split("=")
      term = line[0].lower().strip()
      value = line[1].strip()
      print "%-15s =   %s" % (term, value)
      if term == 'batch_size':
        self.batch_size = int(value)
      elif term == "maxlength":
        self.maxlength = int(value)
      elif term == "filters":
        self.filters = int(value)
      elif term == "hiddendim":
        self.hiddendim = int(value)
      elif term == "num_epochs":
        self.num_epochs = int(value)
      elif term == "rep_dim":
        self.rep_dim = int(value)
        self.offset = self.rep_dim/2 -1
        self.block_size = 0.1528
        self.space_size = 3.0
        self.unit_size = self.space_size / self.rep_dim
      elif term == "directory":
        self.Directory = value
      elif term == "traindata":
        self.TrainData = value
      elif term == "evaldata":
        self.EvalData = value
      elif term == "raweval":
        self.RawEval = value
      elif term == "dropout":
        self.dropout = float(value)
      elif term == "multicells":
        self.multicells = int(value)
      elif term == "vocabfile":
        self.VocabFile = value
