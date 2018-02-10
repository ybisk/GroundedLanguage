import yaml
import sys
import os
import importlib
import tensorflow as tf
from TFLibraries.config import Config

import data

""" Parameters """
config = Config(yaml.load(open(sys.argv[1])))
tf.set_random_seed(config.random_seed)

""" TensorBoard Directories """
name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
summary_path = './summary/{}'.format(name)
checkpoint_path = './summary/{}.ckpt'.format(name)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

config.summary_path = summary_path
config.checkpoint_path = checkpoint_path

""" Read Data """
Training = data.Data(config, train=True)
Development = data.Data(config, train=False, Training=Training)

Model = getattr(importlib.import_module(config.model), 'Model')
model = Model(Training, Development, None, config)

train = True
if train:
  model.train(epochs=config.epochs)
else:
  model.interactive()
