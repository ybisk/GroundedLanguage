import os

class Config(object):
  def __init__(self, params):
    self.params = params
    self.model = self._check_default("model", 'Language_Pipeline')
    self.load_model = self._check_default("load_model", None)
    self.random_seed = self._check_default("random_seed", 20170406)
    self.gpu_list = self.get_gpus()

    # Training curriculum
    self.modes = self._check_default("modes", ['vision', 'copy', 'rl'])
    self.epochs = self._check_default("epochs", [3,3,100])

    # Data
    self.training = self._check_default("training", 'data/trainset.json.gz')
    self.evaluation = self._check_default("evaluation", 'data/devset.json.gz')
    self.predict_source = self._check_default("predict_source", False)
    self.operations = self._check_default("operations", ["one"])
    self.operation_weights = self._check_default("operation_weights", [1.0])
    self.batch_size = self._check_default("batch_size", 32)

    # Language
    self.max_length = self._check_default("max_length", 30)
    self.oov = self._check_default("oov", 20)
    self.txt_dim = self._check_default("txt_dim", 64)
    self.txt_h_dim = self._check_default("txt_h_dim", 1024)
    self.num_ops = self._check_default("num_ops", 32)
    self.dropout = self._check_default("dropout", 1.0)

    # Image
    self.rep_dim = self._check_default("rep_dim", 100)
    self.rep_dim_y = self._check_default("rep_dim_y", 1)
    self.hidden_dim = self._check_default("hidden_dim", 128)
    self.pixel_dim = self._check_default("pixel_dim", 16)
    self.conv_layers = self._check_default("conv_layers", 2)
    self.kernel_size = self._check_default("kernel_size", 5)
    self.kernel_size_y = self._check_default("kernel_size_y", 1)
    self.batch_norm = self._check_default("batch_norm", True)
    self.non_linearity = self._check_default("non_linearity", 'lrelu')
    self.attention_size = self._check_default("attention_size", 10)

    # Extra
    self.regularizer = self._check_default("regularizer", None)
    self.exploration = self._check_default("exploration", 0.3)
    self.exhaustive = self._check_default("exhaustive", False)

    # Evaluation
    self.load_source_model = self._check_default("load_source_model", None)
    self.load_target_model = self._check_default("load_target_model", None)

  def _check_default(self, name, default):
    if name in self.params:
      new_v = self.params[name]
    else:
      new_v = default
    print "{:<25} {:<25}".format(name, new_v)
    return new_v

  def get_gpus(self):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or \
            os.environ['CUDA_VISIBLE_DEVICES'] == '':
      return [0]
    else:
      gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
      num_gpus = len(gpus)
      return range(num_gpus)
