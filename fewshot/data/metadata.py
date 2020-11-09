from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


def get_metadata(dataset_name):
  """Gets dataset metadata."""
  if dataset_name == 'mini-imagenet':
    mean_pix = np.array(
        [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]])
    std_pix = np.array(
        [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
    return {
        'train_split': 'train_phase_train',
        'val_split': 'train_phase_val',
        'test_split': 'train_phase_test',
        'train_fs_split': 'train_phase_train',
        'val_fs_split': 'val',
        'test_fs_split': 'test',
        'mean_pix': mean_pix,
        'std_pix': std_pix,
        'image_size': 84,
        'crop_size': 92,
        'random_crop': True,
        'random_flip': True,
        'random_color': False
    }
  elif dataset_name == 'tiered-imagenet':
    return {
        'train_split': 'train_phase_train',
        'val_split': 'train_phase_val',
        'test_split': 'train_phase_test',
        'train_fs_split': 'train_phase_train',
        'val_fs_split': 'val',
        'test_fs_split': 'test',
        'mean_pix': None,
        'std_pix': None,
        'image_size': 84,
        'crop_size': 92,
        'random_crop': True,
        'random_flip': True,
        'random_color': False
    }
  elif dataset_name == 'omniglot':
    return {
        'train_split': None,  # Does not support supervised pretrain here.
        'val_split': None,  # Does not support supervised pretrain here.
        'test_split': None,  # Does not support supervised pretrain here.
        'train_fs_split': 'train',
        'val_fs_split': 'val',
        'test_fs_split': 'test',
        'mean_pix': None,
        'std_pix': None,
        'image_size': 28,
        'crop_size': 32,
        'random_crop': True,
        'random_flip': False,
        'random_color': False
    }
  elif dataset_name == 'mnist':
    return {
        'train_split': 'train',
        'val_split': 'val',
        'test_split': 'test',
        'train_fs_split': None,  # Does not support few-shot here.
        'val_fs_split': None,  # Does not support few-shot here.
        'test_fs_split': None,  # Does not support few-shot here.
        'mean_pix': None,
        'std_pix': None,
        'image_size': 28,
        'crop_size': 32,
        'random_crop': True,
        'random_flip': False,
        'random_color': False
    }
  else:
    raise Exception('Unknown dataset {}'.format(dataset_name))
