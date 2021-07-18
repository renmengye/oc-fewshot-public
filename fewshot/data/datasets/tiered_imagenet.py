"""Tiered-imagenet dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# from collections import OrderedDict

import cv2
import os
import numpy as np
import pickle as pkl

from fewshot.data.datasets.pickle_cache_dataset import PickleCacheDataset
from fewshot.data.registry import RegisterDataset
from fewshot.data.compress_tiered_imagenet import decompress
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterDataset('tiered-imagenet')
class TieredImageNetDataset(PickleCacheDataset):

  def __init__(self, folder, split, png=True):
    """Creates a dataset with pickle cache.

    Args:
      folder: String. Folder path.
      split: String. Split name.
    """
    self._png = png
    super(TieredImageNetDataset, self).__init__(folder, split)
    self._labels_general = self._data['labels_general']
    self._label_general_str = self._data['label_general_str']
    self._cls_dict = self.get_cls_dict()
    self._cls_hierarchy_dict = self.get_hierarchy_dict()

  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self._folder, self._split)
    cache_path_labels = cache_path + "_labels.pkl"
    cache_path_images = cache_path + "_images.npy"
    return cache_path_labels, cache_path_images

  def read_cache(self, cache_path):
    cache_path_labels = cache_path[0]
    cache_path_images = cache_path[1]
    png_pkl = cache_path_images[:-4] + '_png.pkl'
    # Decompress images.
    if not self._png and not os.path.exists(cache_path_images):
      if os.path.exists(png_pkl):
        decompress(cache_path_images, png_pkl)
    # print(cache_path_labels, cache_path_images)
    if os.path.exists(cache_path_labels) and os.path.exists(cache_path_images):
      log.info("Read cached labels from {}".format(cache_path_labels))
      try:
        with open(cache_path_labels, "rb") as f:
          data = pkl.load(f, encoding='bytes')
          labels = data[b"label_specific"]
          label_str = data[b"label_specific_str"]
          labels_general = data[b"label_general"]
          label_general_str = data[b"label_general_str"]
      except:  # NOQA
        with open(cache_path_labels, "rb") as f:
          data = pkl.load(f)
          labels = data["label_specific"]
          label_str = data["label_specific_str"]
          labels_general = data["label_general"]
          label_general_str = data["label_general_str"]
      if self._png:
        with open(png_pkl, 'rb') as f:
          images = pkl.load(f, encoding='bytes')
      else:
        assert False, "Please use PNG mode"
        images = np.load(cache_path_images)
      return {
          "images": images,
          "labels": labels,
          "label_str": label_str,
          "labels_general": labels_general,
          "label_general_str": label_general_str
      }
    else:
      assert False

  def _read_dataset(self):
    # TODO: get this done.
    raise NotImplementedError()

  def get_ids(self):
    return np.arange(len(self.images))

  def get_images(self, inds):
    """Gets an image or a batch of images."""
    if self._png:
      images = None
      if type(inds) == int:
        images = cv2.imdecode(self.images[inds], 1)
      else:
        for ii, item in enumerate(inds):
          im = cv2.imdecode(self.images[item], 1)
          if images is None:
            images = np.zeros(
                [len(inds), im.shape[0], im.shape[1], im.shape[2]],
                dtype=im.dtype)
          images[ii] = im
      return images
    else:
      return self.images[inds]

  def get_labels(self, inds):
    """Gets the label of an image or a batch of images."""
    return self.labels[inds]

  def get_hierarchy_dict(self):
    """Computes class hierarchy dictionary."""
    hdict = {}
    for l_g, l_s in zip(self.labels_general, self.labels):
      if l_g not in hdict:
        hdict[l_g] = [l_s]
      else:
        if l_s not in hdict[l_g]:
          hdict[l_g].append(l_s)

    for k in hdict:
      hdict[k] = sorted(hdict[k])
    # for i, l in enumerate(self.labels_general):
    #   a = l
    #   if a not in hdict:
    #     hdict[a] = [i]
    #   else:
    #     hdict[a].append(i)
    # print(hdict)
    return hdict

  @property
  def labels_general(self):
    return self._labels_general

  @property
  def label_general_str(self):
    return self._label_general_str

  @property
  def cls_dict(self):
    return self._cls_dict

  @property
  def cls_hierarchy_dict(self):
    return self._cls_hierarchy_dict


if __name__ == '__main__':
  TieredImageNetDataset('./data/tiered-imagenet-orig', 'test')
  TieredImageNetDataset('./data/tiered-imagenet-orig', 'val')
  TieredImageNetDataset('./data/tiered-imagenet-orig', 'train')
