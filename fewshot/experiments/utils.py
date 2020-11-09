"""
Training utilities.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import os
import sys
import tensorflow as tf
import time

from google.protobuf.text_format import Merge, MessageToString

from fewshot.data.data_factory import get_dataset


class ExperimentLogger():

  def __init__(self, writer):
    self._writer = writer

  def log(self, name, niter, value, family=None):
    tf.summary.scalar(name, float(value), step=niter)

  def flush(self):
    """Flushes results to disk."""
    self._writer.flush()

  def close(self):
    """Closes writer."""
    self._writer.close()


def save_config(config, save_folder):
  """Saves configuration to a file."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "config.prototxt")
  with open(config_file, "w") as f:
    f.write(MessageToString(config))
  cmd_file = os.path.join(save_folder, "cmd-{}.txt".format(int(time.time())))
  if not os.path.exists(cmd_file):
    with open(cmd_file, "w") as f:
      f.write(' '.join(sys.argv))


def get_config(config_file, config_cls):
  """Reads configuration."""
  config = config_cls()
  Merge(open(config_file).read(), config)
  return config


def get_data_fs(env_config, load_train=False):
  """Gets few-shot dataset."""
  train_split = env_config.train_fs_split
  if train_split is None or (train_split == env_config.train_split and
                             not load_train):
    data_train_fs = None
  else:
    data_train_fs = get_dataset(env_config.dataset, env_config.data_folder,
                                env_config.train_fs_split)
  if env_config.val_fs_split is None:
    data_val_fs = None
  else:
    data_val_fs = get_dataset(env_config.dataset, env_config.data_folder,
                              env_config.val_fs_split)
  if env_config.test_fs_split is None:
    data_test_fs = None
  else:
    data_test_fs = get_dataset(env_config.dataset, env_config.data_folder,
                               env_config.test_fs_split)
  return {
      'train_fs': data_train_fs,
      'val_fs': data_val_fs,
      'test_fs': data_test_fs,
      'metadata': env_config
  }


def latest_file(folder, prefix):
  """Query the most recent checkpoint."""
  list_of_files = glob.glob(os.path.join(folder, prefix + '*'))
  if len(list_of_files) == 0:
    return None
  latest_file = max(list_of_files, key=lambda f: int(f.split('-')[-1]))
  return latest_file
