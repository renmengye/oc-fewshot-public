"""Iterator for semi-supervised episodes.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.iterators.episode_iterator import EpisodeIterator
from fewshot.data.registry import RegisterIterator


@RegisterIterator('semisupervised-episode')
class SemiSupervisedEpisodeIterator(EpisodeIterator):
  """Generates semi-supervised episodes. Note that this class doesn't support a
  fixed label/unlabel split on the image level."""

  def __init__(self,
               dataset,
               sampler,
               batch_size,
               nclasses,
               nquery,
               expand=False,
               preprocessor=None,
               episode_processor=None,
               fix_unknown=False,
               maxlen=-1,
               prefetch=True,
               **kwargs):
    super(SemiSupervisedEpisodeIterator, self).__init__(
        dataset,
        sampler,
        batch_size,
        nclasses,
        nquery,
        expand=expand,
        preprocessor=preprocessor,
        episode_processor=episode_processor,
        fix_unknown=fix_unknown,
        maxlen=maxlen,
        prefetch=prefetch,
        **kwargs)
    assert 'label_ratio' in kwargs, 'Must specify label ratio'
    self._label_ratio = kwargs['label_ratio']
    assert fix_unknown, 'Must fix unknown token for semi-supervised task.'

  def process_one(self, collection):
    """Process one episode.

    Args:
      Collection dictionary that contains the following keys:
        support: np.ndarray. Image ID in the support set.
        flag: np.ndarray. Binary flag indicating whether it is labeled (1) or
          unlabeled (0).
        query: np.ndarray. Image ID in the query set.
    """
    s, flag, q = collection['support'], collection['flag'], collection['query']
    del collection['support']
    del collection['query']
    del collection['flag']
    dataset = self.dataset
    nclasses = self.nclasses
    img_s = dataset.get_images(s)
    lbl_s = np.array(collection['support_label'])
    del collection['support_label']
    T = self.maxlen

    # Mask off unlabeled set.
    labeled = flag == 1
    unlabeled = flag == 0
    lbl_s_l = lbl_s[labeled]
    lbl_s_u = lbl_s[unlabeled]

    # Note numpy does not give the desired behavior here.
    # lbl_map, lbl_s_l = np.unique(lbl_s_l, return_inverse=True)
    lbl_map, lbl_s_l = tf.unique(lbl_s_l)

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

    # Find distractors.
    lbl_s_eq = tf.cast(tf.equal(tf.expand_dims(lbl_s, 1), lbl_map), tf.float32)
    distractor_flag = tf.cast(1.0 - tf.reduce_sum(lbl_s_eq, [1]), tf.int32)

    # Re-indexed labels.
    lbl_s[labeled] = lbl_s_l
    lbl_s[unlabeled] = query_np(lbl_s_u)

    # Label fed into the network.
    lbl_s_masked = np.copy(lbl_s)
    lbl_s_masked[unlabeled] = nclasses

    # We assumed fix unknown.
    # Make the first appearing item to be unknown in groundtruth.
    lbl_s_np = np.copy(lbl_s)
    lbl_s_np2 = np.copy(lbl_s_np)
    lbl_s_np2[unlabeled] = -1
    lbl_s_gt = np.zeros([len(lbl_s_np)], dtype=np.int32)
    cummax = np.maximum.accumulate(lbl_s_np2)
    lbl_s_gt[0] = nclasses
    # Labeled to be trained as target.
    cond = lbl_s_np[1:] > cummax[:-1]
    lbl_s_gt[1:] = np.where(cond, nclasses, lbl_s_np[1:])

    if self.nquery > 0:
      img_q = dataset.get_images(q)
      lbl_q = collection['query_label']
      del collection['query_label']
      lbl_q = query_tf(lbl_q)
    else:
      img_q = None
      lbl_q = None
    epi = {
        'x_s': self.pad_x(img_s, T),
        'y_s': self.pad_y(lbl_s_masked, T),
        'y_gt': self.pad_y(lbl_s_gt, T),
        'y_dis': self.pad_y(distractor_flag, T),
        'y_full': self.pad_y(lbl_s, T),
        'flag_s': self.get_flag(lbl_s, T)
    }
    if self.nquery > 0:
      assert False, 'Not supported'

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
