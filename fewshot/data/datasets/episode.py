"""Episode object."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf


class Episode(object):

  def __init__(self,
               train_images,
               train_labels,
               test_images,
               test_labels,
               train_groundtruth=None,
               train_distractor_flag=None,
               train_labels_full=None,
               train_flag=None,
               test_flag=None):
    """An episode object.

    Args:
      train_images: np.ndarray. Support set images.
      train_labels: np.ndarray. Support set integer labels.
      test_images: np.ndarray. Query set images.
      test_labels: np.ndarray. Query set integer labels.
      train_groundtruth: np.ndarray. Groundtruth for loss, default is the same
        as `train_labels`.
      train_distractor_flag: np.ndarray. Binary indicator whether the timestep
        is a distractor.
      train_labels_full: np.ndarray. Support set integer labels without semi-
        supervised masks.
      train_flag: np.ndarray. Binary indicator whether the timestep is valid.
      test_flag: np.ndarray. Binary indicator whether the timestep is valid.
    """
    self._train_images = train_images
    self._test_images = test_images
    self._train_labels = train_labels
    self._test_labels = test_labels
    self._train_flag = train_flag
    self._test_flag = test_flag
    if train_distractor_flag is None:
      self._train_distractor_flag = np.zeros_like(train_labels)
    else:
      self._train_distractor_flag = train_distractor_flag
    if train_groundtruth is None:
      self._train_groundtruth = train_labels
    else:
      self._train_groundtruth = train_groundtruth
    if train_labels_full is None:
      self._train_labels_full = train_labels
    else:
      self._train_labels_full = train_labels_full

  @property
  def train_images(self):
    return self._train_images

  @property
  def train_labels(self):
    return self._train_labels

  @property
  def test_images(self):
    return self._test_images

  @property
  def test_labels(self):
    return self._test_labels

  @property
  def train_flag(self):
    return self._train_flag

  @property
  def test_flag(self):
    return self._test_flag

  @property
  def train_distractor_flag(self):
    return self._train_distractor_flag

  @property
  def train_groundtruth(self):
    return self._train_groundtruth

  @property
  def train_labels_full(self):
    return self._train_labels_full


def pad(images, *labels, maxlen=-1):
  if maxlen == -1:
    max_train_len = max([int(i.shape[0]) for i in images])
  else:
    max_train_len = maxlen
  pad = [max_train_len - int(i.shape[0]) for i in images]
  if all([p == 0 for p in pad]):  # If all zeros.
    return images, labels, [
        tf.ones([int(l.shape[0])], dtype=tf.int32) for l in labels[0]
    ]
  pad_images = []
  # Pad same images.
  for p, i in zip(pad, images):
    pad_images_list = [i]
    counter = p
    while counter > 0:
      remain = i[:counter]
      pad_images_list.append(remain)
      counter -= remain.shape[0]
    pad_images.append(tf.concat(pad_images_list, axis=0))

  # Pad zeros.
  pad_labels = []
  for labels_ in labels:
    ll = [int(l.shape[0]) for l in labels_]
    m = max_train_len
    pad_labels_ = [
        tf.concat([l, tf.zeros([m - ls], dtype=tf.int32)], axis=0)
        for l, ls in zip(labels_, ll)
    ]
    pad_labels.append(pad_labels_)

  # Pad zeros.
  flag = [
      tf.concat(
          [tf.ones([ls], dtype=tf.int32),
           tf.zeros([m - ls], dtype=tf.int32)],
          axis=0) for l, ls in zip(labels[0], ll)
  ]
  return pad_images, pad_labels, flag


def merge_episodes(*episodes, maxlen=-1):
  train_images = [e.train_images for e in episodes]
  train_labels = [e.train_labels for e in episodes]
  test_images = [e.test_images for e in episodes]
  test_labels = [e.test_labels for e in episodes]
  train_gt = [e.train_groundtruth for e in episodes]
  train_dis = [e.train_distractor_flag for e in episodes]
  train_labels_full = [e.train_labels_full for e in episodes]
  train_images_, (train_labels_, train_gt_, train_dis_,
                  train_labels_full_), train_flag_ = pad(
                      train_images,
                      train_labels,
                      train_gt,
                      train_dis,
                      train_labels_full,
                      maxlen=maxlen)
  if test_images[0] is not None:
    test_images_, test_labels_, test_flag_ = pad(
        test_images, test_labels, maxlen=maxlen)
    test_labels_ = test_labels_[0]
  else:
    test_images_ = test_images
    test_labels_ = test_labels
    test_flag_ = None

  def _stack(x):
    return tf.stack(
        x, axis=0) if (x is not None and x[0] is not None) else None

  return Episode(
      _stack(train_images_),
      _stack(train_labels_),
      _stack(test_images_),
      _stack(test_labels_),
      train_groundtruth=_stack(train_gt_),
      train_distractor_flag=_stack(train_dis_),
      train_labels_full=_stack(train_labels_full_),
      train_flag=_stack(train_flag_),
      test_flag=_stack(test_flag_))


def expand_episode(episode):

  def _expand(x):
    return tf.expand_dims(x, axis=0) if x is not None else None

  episode._train_images = _expand(episode._train_images)
  episode._test_images = _expand(episode._test_images)
  episode._train_labels = _expand(episode._train_labels)
  episode._test_labels = _expand(episode._test_labels)
  episode._train_groundtruth = _expand(episode._train_groundtruth)
  episode._train_distractor_flag = _expand(episode._train_distractor_flag)
  episode._train_labels_full = _expand(episode._train_labels_full)
  episode._train_flag = _expand(episode._train_flag)
  episode._test_flag = _expand(episode._test_flag)
  return episode
