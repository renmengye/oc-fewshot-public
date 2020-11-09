"""Iterators for few-shot episode.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.iterators.iterator import Iterator
from fewshot.data.registry import RegisterIterator


@RegisterIterator('episode')
class EpisodeIterator(Iterator):
  """Generates lifelong episodes."""

  def __init__(self,
               dataset,
               sampler,
               batch_size,
               nclasses,
               nquery,
               preprocessor=None,
               episode_processor=None,
               fix_unknown=False,
               maxlen=-1,
               prefetch=True,
               **kwargs):
    """Creates a lifelong learning data iterator.

    Args:
      dataset: A dataset object. See `fewshot/data/dataset.py`.
      sampler: A sampler object. See `fewshot/data/sampler.py`.
      batch_size: Number of episodes together.
      nclasses: Total number of new classes.
      nquery: Query period, number of query images per class.
      fix_unknown: Whether the unknown token is k+1 or fixed at K+1.
      maxlen: Maximum length of the sequence.
      preprocessor: Image preprocessor.
    """
    self._dataset = dataset
    self._sampler = sampler
    sampler.set_dataset(dataset)
    self._nclasses = nclasses
    self._batch_size = batch_size
    self._nclasses = nclasses
    self._nquery = nquery
    assert batch_size >= 1
    self._preprocessor = preprocessor
    self._episode_processor = episode_processor
    self._kwargs = kwargs
    self._maxlen = maxlen
    assert maxlen > 0
    self._fix_unknown = fix_unknown
    self._prefetch = prefetch
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def process_one(self, collection):
    """Process one episode.

    Args:
      Collection dictionary that contains the following keys:
        support: np.ndarray. Image ID in the support set.
        query: np.ndarray. Image ID in the query set.
    """
    s, q = collection['support'], collection['query']
    del collection['support']
    del collection['query']
    dataset = self.dataset
    nclasses = self.nclasses
    img_s = dataset.get_images(s)
    lbl_s = collection['support_label']
    del collection['support_label']
    T = self.maxlen
    lbl_map, lbl_s = tf.unique(lbl_s)

    # np.unique returns a sorted lbl_map, but not according to appear order.
    # tf.unique returns according to appear order.
    # lbl_map, lbl_s = np.unique(lbl_s, return_index=True)
    # print('before', lbl_s2, lbl_map2)
    # print('after', lbl_s, lbl_map)
    # assert False

    def query_tf(x):
      x = tf.expand_dims(x, 1)  # [T, 1]
      x_eq = tf.cast(tf.equal(x, lbl_map), tf.float32)  # [T, N]
      x_valid = tf.reduce_sum(x_eq, [1])  # [T]

      # Everything that has not been found -> fixed unknown.
      # This means it's a distractor.
      x = tf.cast(tf.argmax(x_eq, axis=1), tf.float32)
      x = x_valid * x + (1 - x_valid) * nclasses
      x = tf.cast(x, tf.int32)
      return x

    def query_np(x):
      x = np.expand_dims(x, 1)  # [T, 1]
      x_eq = np.equal(x, lbl_map).astype(np.float32)  # [T, N]
      x_valid = np.sum(x_eq, axis=1)  # [T]

      # Everything that has not been found -> fixed unknown.
      # This means it's a distractor.
      x = np.argmax(x_eq, axis=1).astype(np.float32)
      x = x_valid * x + (1 - x_valid) * nclasses
      x = x.astype(np.int32)
      return x

    # Generates the groundtruth.
    if self._fix_unknown:
      # Make the first appearing image have the fixed UNK class.
      lbl_s_np = lbl_s.numpy()
      lbl_s_gt = np.zeros([len(lbl_s_np)], dtype=np.int32)
      cummax = np.maximum.accumulate(lbl_s_np)
      lbl_s_gt[0] = nclasses
      cond = lbl_s_np[1:] > cummax[:-1]
      lbl_s_gt[1:] = np.where(cond, nclasses, lbl_s_np[1:])
    else:
      lbl_s_gt = lbl_s

    if self.nquery > 0:
      img_q = dataset.get_images(q)
      lbl_q = collection['query_label']
      del collection['query_label']
      # lbl_q = dataset.get_labels(q)
      lbl_q = query_tf(lbl_q)
    else:
      img_q = None
      lbl_q = None
    epi = {
        'x_s': self.pad_x(img_s, T),
        'y_s': self.pad_y(lbl_s, T),
        'y_gt': self.pad_y(lbl_s_gt, T),
        'y_dis': tf.zeros([T], dtype=lbl_s.dtype),
        'y_full': self.pad_y(lbl_s, T),
        'flag_s': self.get_flag(lbl_s, T)
    }
    if self.nquery > 0:
      T2 = self.nquery * self.nclasses
      epi['x_q'] = self.pad_x(img_q, T2)
      epi['y_q'] = self.pad_y(lbl_q, T2)
      epi['flag_q'] = self.get_flag(lbl_q, T2)
      if 'stage_id_q' in collection:
        epi['stage_id_q'] = self.pad_y(collection['stage_id_q'], T2)
        del collection['stage_id_q']

    # For remaining additional info.
    for k in collection:
      epi[k] = self.pad_y(collection[k], T)

    if self.episode_processor is not None:
      epi = self.episode_processor(epi)
    return epi

  def _next(self):
    """Next example."""
    collection = self.sampler.sample_collection(self.nclasses, self.nquery,
                                                **self.kwargs)
    return self.process_one(collection)

  def pad_x(self, x, maxlen):
    """Pad image sequence."""
    T = x.shape[0]
    return np.pad(x, [[0, maxlen - T], [0, 0], [0, 0], [0, 0]], mode='reflect')

  def pad_y(self, y, maxlen):
    """Pad label sequence."""
    T = y.shape[0]
    return np.pad(y, [0, maxlen - T], mode='constant', constant_values=0)

  def get_flag(self, y, maxlen):
    """Get valid flag."""
    return (np.arange(maxlen) < y.shape[0]).astype(np.int32)

  def __iter__(self):
    return self._tf_dataset_iter

  def get_generator(self):
    """Gets generator function, for tensorflow Dataset object."""
    while True:
      yield self._next()

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dummy = self._next()
    self.sampler.reset()
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict([(k, tf.shape(dummy[k])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)

    def preprocess(data):
      data['x_s'] = self.preprocessor(data['x_s'])
      if self.nquery > 0:
        data['x_q'] = self.preprocessor(data['x_q'])
      return data

    ds = ds.map(preprocess)
    ds = ds.batch(self.batch_size)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  def reset(self):
    """Resets sampler."""
    self.sampler.reset()

  @property
  def kwargs(self):
    """Additional parameters for the sampler."""
    return self._kwargs

  @property
  def preprocessor(self):
    """Image preprocessor."""
    return self._preprocessor

  @property
  def episode_processor(self):
    """Episode processor."""
    return self._episode_processor

  @property
  def sampler(self):
    """Episode sampler."""
    return self._sampler

  @property
  def dataset(self):
    """Dataset source."""
    return self._dataset

  @property
  def nclasses(self):
    """Number of classes per episode."""
    return self._nclasses

  @property
  def nquery(self):
    """Number of query examples per class per episode."""
    return self._nquery

  @property
  def batch_size(self):
    """Number of episodes."""
    return self._batch_size

  @property
  def maxlen(self):
    """Max length of the sequence."""
    return self._maxlen

  @property
  def tf_dataset(self):
    """TF dataset API."""
    return self._tf_dataset
