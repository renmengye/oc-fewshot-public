"""Iterator for regular mini-batches.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import threading
import tensorflow as tf


class MinibatchIterator(object):
  """Generates mini-batches for pretraining."""

  def __init__(self,
               dataset,
               sampler,
               batch_size,
               prefetch=True,
               preprocessor=None):
    self._dataset = dataset
    self._preprocessor = preprocessor
    self._sampler = sampler
    sampler.set_num(dataset.get_size())
    self._mutex = threading.Lock()
    assert batch_size > 0, 'Need a positive number for batch size'
    self._batch_size = batch_size
    self._prefetch = prefetch
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def __iter__(self):
    return self

  def reset(self):
    self.sampler.reset()
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def get_generator(self):
    while True:
      # TODO: change the mutex lock here to TF dataset API.
      self._mutex.acquire()
      try:
        idx = self.sampler.sample_collection(self.batch_size)
      finally:
        self._mutex.release()
      if idx is None:
        return
      assert idx is not None
      x = self.dataset.get_images(idx)
      y = self.dataset.get_labels(idx)
      yield {'x': x, 'y': y}

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dtype_dict = {
        'x': tf.uint8,
        'y': tf.int32,
    }
    shape_dict = {
        'x': tf.TensorShape([None, None, None, None]),
        'y': tf.TensorShape([None]),
    }
    N = self.dataset.get_size()
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)

    def preprocess(data):
      data['x'] = self.preprocessor(data['x'])
      return data

    ds = ds.map(preprocess)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  @property
  def dataset(self):
    """Dataset object."""
    return self._dataset

  @property
  def preprocessor(self):
    """Data preprocessor."""
    return self._preprocessor

  @property
  def sampler(self):
    """Mini-batch sampler."""
    return self._sampler

  @property
  def step(self):
    """Number of steps."""
    return self.sampler._step

  @property
  def epoch(self):
    """Number of epochs."""
    return self.sampler._epoch

  @property
  def batch_size(self):
    """Batch size."""
    return self._batch_size

  @property
  def tf_dataset(self):
    return self._tf_dataset

  @property
  def cycle(self):
    return self._cycle

  @property
  def shuffle(self):
    return self._shuffle
