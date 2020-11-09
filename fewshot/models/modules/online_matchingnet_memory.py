"""Online matching net -- an online version of the nearest neighbor algorithm.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.example_memory import ExampleMemory
from fewshot.models.registry import RegisterModule

INF = 1e6


@RegisterModule('online_matchingnet_memory')
@RegisterModule('matchingnet_memory')  # Legacy name
class OnlineMatchingNetMemory(ExampleMemory):

  def compute_cosine_sim(self, a, b):
    """Computes cosine similarity."""
    ab = tf.matmul(a, b, transpose_b=True)  # [B, K1, K2]
    anorm = tf.maximum(tf.sqrt(tf.reduce_sum(a**2, [-1])), 1e-7)  # [B, K1]
    bnorm = tf.maximum(tf.sqrt(tf.reduce_sum(b**2, [-1])), 1e-7)  # [B, K2]
    return ab / tf.expand_dims(anorm, 2) / tf.expand_dims(bnorm, 1)

  def infer(self, x, t, storage, label):
    """Infer cluster ID. Either goes into one of the existing cluster
    or become a new cluster. This procedure is for prediction purpose.

    Args:
      x: Input. [B, D]

    Returns:
      logits: Cluster logits. [B, M]
      new_prob: New cluster probability. [B]
    """
    B = x.shape[0]
    K = self.max_classes
    if tf.equal(t, 0):
      return tf.zeros([B, K + 1],
                      dtype=self.dtype), tf.zeros([B], dtype=self.dtype) + INF
    storage_ = storage[:, :t, :]
    label_ = label[:, :t]
    x_ = tf.expand_dims(x, 1)  # [B, 1, D]

    if self._similarity == "cosine":
      logits = tf.squeeze(self.compute_cosine_sim(x_, storage_), 1)  # [B, M]
      kprob = tf.nn.softmax(logits * 7.5)  # [B, M]
    elif self._similarity == "euclidean":
      logits = -tf.squeeze(self.compute_euclidean_dist_sq(x_, storage_),
                           1)  # [B, M]
      kprob = tf.nn.softmax(logits)

    max_logits = tf.reduce_max(logits, [-1])  # [B]
    clabel_onehot = tf.one_hot(label_, self.unknown_id + 1)  # [B, M, C]
    # [B, M, 1] * [B, M, C] = [B, C]
    cprob = tf.reduce_sum(tf.expand_dims(kprob, -1) * clabel_onehot, [1])
    cprob = tf.maximum(cprob, 1e-6)  # Delta.
    cprob.set_shape([B, K + 1])
    new = (self._beta - max_logits) / self._gamma  # [B]
    # remain = (max_logits - self._beta) / self._gamma  # [B]
    return tf.math.log(cprob), new
