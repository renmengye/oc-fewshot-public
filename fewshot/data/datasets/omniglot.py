"""Omniglot dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import numpy as np
import os

from fewshot.data.datasets.pickle_cache_dataset import PickleCacheDataset
from fewshot.data.registry import RegisterDataset
from fewshot.utils import logger

log = logger.get()


def get_image_folder(folder, split_def, split):
  if split_def == 'lake':
    if split == 'train_all':
      folder_ = os.path.join(folder, 'images_background')
    elif split == 'train':
      folder_ = os.path.join(folder, 'images_background_train')
    elif split == 'val':
      folder_ = os.path.join(folder, 'images_background_val')
    elif split == 'test':
      folder_ = os.path.join(folder, 'images_evaluation')
  elif split_def == 'vinyals':
    folder_ = os.path.join(folder, 'images_all')
  elif split_def == 'ren':
    folder_ = os.path.join(folder, 'images_all')
  return folder_


def get_vinyals_split_file(split):
  curdir = os.path.dirname(os.path.realpath(__file__))
  split_file = os.path.join(curdir, '../omniglot_split_vinyals',
                            '{}.txt'.format(split))
  return split_file


def get_ren_split_file(split):
  curdir = os.path.dirname(os.path.realpath(__file__))
  split_file = os.path.join(curdir, '../omniglot_split_new',
                            '{}.txt'.format(split))
  return split_file


def read_lake_split(folder, aug_90=False):
  """Reads dataset from folder."""
  subfolders = os.listdir(folder)
  label_idx = []
  label_str = []
  data = []
  for sf in subfolders:
    sf_ = os.path.join(folder, sf)
    img_fnames = os.listdir(sf_)
    for character in img_fnames:
      char_folder = os.path.join(sf_, character)
      img_list = os.listdir(char_folder)
      for img_fname in img_list:
        fname_ = os.path.join(char_folder, img_fname)
        img = cv2.imread(fname_)
        # Shrink images.
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.minimum(255, np.maximum(0, img))
        img = 255 - img[:, :, 0:1]
        if aug_90:
          M = cv2.getRotationMatrix2D((14, 14), 90, 1)
          dst = img
          for ii in range(4):
            dst = cv2.warpAffine(dst, M, (28, 28))
            data.append(np.expand_dims(np.expand_dims(dst, 0), 3))
            label_idx.append(len(label_str) + ii)
        else:
          img = np.expand_dims(img, 0)
          data.append(img)
          label_idx.append(len(label_str))

      if aug_90:
        for ii in range(4):
          label_str.append(sf + '_' + character + '_' + str(ii))
      else:
        label_str.append(sf + '_' + character)
  print('Number of classes {}'.format(len(label_str)))
  print('Number of images {}'.format(len(data)))
  images = np.concatenate(data, axis=0)
  labels = np.array(label_idx, dtype=np.int32)
  lab_maxel_str = label_str
  return images, labels, label_str


def read_vinyals_split(folder, split_file, aug_90=False):
  """Reads dataset from a folder with a split file."""
  lines = open(split_file, 'r').readlines()
  lines = map(lambda x: x.strip('\n\r'), lines)
  label_idx = []
  label_str = []
  data = []
  for ff in lines:
    char_folder = os.path.join(folder, ff)
    img_list = os.listdir(char_folder)
    for img_fname in img_list:
      fname_ = os.path.join(char_folder, img_fname)
      img = cv2.imread(fname_)
      img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
      img = np.minimum(255, np.maximum(0, img))
      img = 255 - img[:, :, 0:1]
      if aug_90:
        M = cv2.getRotationMatrix2D((14, 14), 90, 1)
        dst = img
        for ii in range(4):
          dst = cv2.warpAffine(dst, M, (28, 28))
          data.append(np.expand_dims(np.expand_dims(dst, 0), 3))
          label_idx.append(len(label_str) + ii)
      else:
        img = np.expand_dims(img, 0)
        data.append(img)
        label_idx.append(len(label_str))
    if aug_90:
      for ii in range(4):
        label_str.append(ff.replace('/', '_') + '_' + str(ii))
    else:
      label_str.append(ff.replace('/', '_'))
  print('Number of classes {}'.format(len(label_str)))
  print('Number of images {}'.format(len(data)))
  images = np.concatenate(data, axis=0)
  labels = np.array(label_idx, dtype=np.int32)
  return images, labels, label_str


# def create_image_split(folder, split_file, split_names, split_ratio):
#   assert sum(split_ratio) == 1.0
#   lines = open(split_file, 'r').readlines()
#   lines = map(lambda x: x.strip('\n\r'), lines)
#   label_idx = []
#   label_str = []
#   data = []
#   for ff in lines:
#     char_folder = os.path.join(folder, ff)
#     img_list = os.listdir(char_folder)
#     for img_fname in img_list:
#       fname_ = os.path.join(char_folder, img_fname)


@RegisterDataset('omniglot')
class OmniglotDataset(PickleCacheDataset):

  def __init__(self, folder, split):
    """Creates an omniglot dataset instance."""
    self._split_def = split.split('_')[0]
    split = '_'.join(split.split('_')[1:])
    super(OmniglotDataset, self).__init__(folder, split)
    self._cls_dict = self.get_cls_dict()
    self._cls_hierarchy_dict = self.get_hierarchy_dict()
    # print('cls', len(self.cls_dict))
    # print('hdict', len(self.cls_hierarchy_dict))

  def _read_dataset(self):
    """Read data from folder or cache."""
    folder, split = self._folder, self._split
    folder = get_image_folder(folder, self._split_def, split)
    if self._split_def == 'lake':
      # aug_90 = self._split.startswith('train')
      # aug_90 = False
      aug_90 = True
      images, labels, label_str = read_lake_split(folder, aug_90=aug_90)
    elif self._split_def == 'vinyals':
      split_file = get_vinyals_split_file(self._split)
      images, labels, label_str = read_vinyals_split(
          folder, split_file, aug_90=True)
    elif self._split_def == 'ren':
      split_file = get_ren_split_file(self._split)
      images, labels, label_str = read_vinyals_split(
          folder, split_file, aug_90=True)
    data = {'images': images, 'labels': labels, 'label_str': label_str}
    return data

  def get_images(self, inds):
    """Gets an image or a batch of images."""
    return self.images[inds]

  def get_labels(self, inds):
    """Gets the label of an image or a batch of images."""
    return self.labels[inds]

  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self.folder,
                              self._split_def + '-' + self.split + '.pkl')
    return cache_path

  def get_cls_dict(self):
    """Computes class dictionary."""
    cls_dict = {}
    for i, l in enumerate(self.labels):
      if l in cls_dict:
        cls_dict[l].append(i)
      else:
        cls_dict[l] = [i]
    return cls_dict

  def get_hierarchy_dict(self):
    """Computes class hierarchy dictionary."""
    assert self._split_def in ['lake', 'ren']
    hdict = {}
    for i, l in enumerate(self.label_str):
      a = l.split('_')[0]
      if a not in hdict:
        hdict[a] = [i]
      else:
        hdict[a].append(i)
    return hdict

  @property
  def cls_dict(self):
    return self._cls_dict

  @property
  def cls_hierarchy_dict(self):
    return self._cls_hierarchy_dict


if __name__ == '__main__':
  OmniglotDataset('./data/omniglot-vinyals', 'ren_train')
  OmniglotDataset('./data/omniglot-vinyals', 'ren_val')
  OmniglotDataset('./data/omniglot-vinyals', 'ren_test')
