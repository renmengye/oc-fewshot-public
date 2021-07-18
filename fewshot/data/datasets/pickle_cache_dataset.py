"""A simple dataset base class that uses pickle to cache the dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import pickle as pkl

from fewshot.data.datasets.dataset import Dataset


class PickleCacheDataset(Dataset):
  """A dataset that uses pickle to cache the data."""

  def __init__(self, folder, split):
    """Creates a dataset with pickle cache.

    Args:
      folder: String. Folder path.
      split: String. Split name.
    """
    super(PickleCacheDataset, self).__init__()
    assert folder is not None
    assert split is not None
    self._folder = folder
    self._split = split
    data = self.read_dataset()
    self._images = data['images']
    self._labels = data['labels']
    self._label_str = data['label_str']
    self._data = data
    self._cls_dict = None

  def __len__(self):
    return self._images.shape[0]

  def read_cache(self, cache_path):
    """Reads dataset from cached pkl file.

    Args:
      cache_path: filename of the cache.

    Returns:
      dict: data dictionary, None if the cache doesn't exist.
    """
    if os.path.exists(cache_path):
      try:
        with open(cache_path, 'rb') as f:
          data = pkl.load(f, encoding='bytes')
          images = data[b'images']
          labels = data[b'labels']
          label_str = data[b'label_str']
      except:  # NOQA
        with open(cache_path, 'rb') as f:
          data = pkl.load(f)
          images = data['images']
          labels = data['labels']
          label_str = data['label_str']
      return {'images': images, 'labels': labels, 'label_str': label_str}
    else:
      return None

  def save_cache(self, cache_path, data):
    """Saves pklz cache."""
    with open(cache_path, 'wb') as f:
      pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self.folder, self.split + '.pkl')
    return cache_path

  def _read_dataset(self):
    raise NotImplementedError()

  def read_dataset(self):
    """Read data from folder or cache."""
    cache_path = self.get_cache_path()
    data = self.read_cache(cache_path)
    if data is None:
      data = self._read_dataset()
      self.save_cache(cache_path, data)
    return data

  def get_size(self):
    """Gets the total number of images."""
    return len(self.images)

  def get_cls_dict(self):
    """Gets class dictionary."""
    if self._cls_dict is None:
      idic = self._label_str
      if type(idic) == dict:
        keys = idic.keys()
      elif type(idic) == list:
        keys = range(len(idic))
      else:
        assert False
      self._cls_dict = {}
      for kk in keys:
        self._cls_dict[kk] = []
      for ii, ll in enumerate(self._labels):
        self._cls_dict[ll].append(ii)
    return self._cls_dict

  @property
  def folder(self):
    """Data folder."""
    return self._folder

  @property
  def split(self):
    """Data split."""
    return self._split

  @property
  def images(self):
    """Image data."""
    return self._images

  @property
  def labels(self):
    """Label data."""
    return self._labels

  @property
  def label_str(self):
    return self._label_str
