class Param:

  batch_size = 512
  maxlength = 40
  filters = 4
  hiddendim = 100
  num_epochs = 12

  rep_dim = 32
  offset = rep_dim/2 -1
  block_size = 0.1528
  space_size = 3.0
  unit_size = space_size / rep_dim

  Directory = '/home/ybisk/GroundedLanguage'
  TrainData = 'Priors/Train.%d.L1.LangAndBlank.20.npz' % rep_dim
  EvalData = 'Priors/Dev.%d.L1.Lang.20.npz' % rep_dim
  RawEval = 'Priors/WithText/Dev.mat.gz'
  #EvalData = 'Priors/Test.Lang.20.npz'
  #RawEval = 'Priors/WithText/Test.mat.gz'

  
  def __init__(self, filename):
    for line in open(filename):
      if len(line) < 2 or line[0] == "#":
        continue
      term, value = line.lower().strip().split("=")
      print "%-15s =   %s" % (term, value)
      if term == "batch_size":
        batch_size = int(value)
      elif term == "maxlength":
        maxlength = int(value)
      elif term == "filters":
        filters = int(value)
      elif term == "hiddendim":
        hiddendim = int(value)
      elif term == "num_epochs":
        num_epochs = int(value)
      elif term == "rep_dim":
        rep_dim = int(value)
        offset = rep_dim/2 -1
        block_size = 0.1528
        space_size = 3.0
        unit_size = space_size / rep_dim
      elif term == "directory":
        Directory = value
      elif term == "traindata":
        TrainData = value
      elif term == "evaldata":
        EvalData = value
      elif term == "raweval":
        RawEval = value
