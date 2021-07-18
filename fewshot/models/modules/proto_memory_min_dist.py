"""A prototypical memory module with minimum distance.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.proto_memory import ProtoMemory
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope

LOGINF = 1e6


@RegisterModule('min_dist_proto_memory')
class MinDistProtoMemory(ProtoMemory):
  """Use an extra cluster to capture the new classes."""

  def __init__(self,
               name,
               dim,
               radius_init,
               max_classes=20,
               fix_unknown=False,
               unknown_id=None,
               similarity="euclidean",
               normalize_feature=False,
               static_beta_gamma=True,
               unknown_logits="radii",
               temp_init=10.0,
               dtype=tf.float32,
               **kwargs):
    super(MinDistProtoMemory, self).__init__(
        name,
        dim,
        max_classes=max_classes,
        fix_unknown=fix_unknown,
        unknown_id=unknown_id,
        similarity=similarity,
        normalize_feature=normalize_feature,
        temp_init=temp_init,
        dtype=dtype)
    self._radius_init = radius_init
    self._unknown_logits = unknown_logits
    if static_beta_gamma:
      with variable_scope(name):
        self._beta = self._get_variable(
            "beta", self._get_constant_init([], radius_init))
        self._gamma = self._get_variable("gamma",
                                         self._get_constant_init([], 1.0))

  def retrieve(self,
               x,
               storage,
               count,
               t,
               beta=None,
               gamma=None,
               add_new=tf.constant(True),
               is_training=tf.constant(True)):
    """See ProtoMemory for documentation."""
    x = tf.expand_dims(x, 1)  # [B, 1, D]
    prototypes = storage
    K = self.max_classes
    dtype = self.dtype
    beta_ = beta if beta is not None else self._beta
    gamma_ = gamma if gamma is not None else self._gamma

    # nonempty = tf.cast(tf.greater(count, 0), dtype)  # [B, K+1]
    # logits = self.compute_logits(x, prototypes)
    # min_dist = tf.reduce_min(-logits, [1])

    nonempty = tf.cast(tf.greater(count, 0), dtype)  # [B, K+1]
    logits = self.compute_logits(x, prototypes)
    cur_num = tf.reduce_sum(
        tf.cast(tf.greater(count, 0), dtype), [1], keepdims=True)  # [B, 1]
    kmask2 = tf.cast(
        tf.less_equal(
            tf.range(storage.shape[1], dtype=tf.int64)[None, :],
            tf.cast(cur_num, tf.int64)), self.dtype)  # [B, K]

    min_dist = tf.reduce_min(-logits, [1])

    if self._unknown_logits == "max":
      out_smax = tf.nn.softmax(logits)
      unk_score = 1.0 - tf.reduce_max(out_smax * kmask2, [-1])  # [B]
      unk_score = tf.minimum(tf.maximum(unk_score, 1e-5), 1 - 1e-3)  # [B]
      log_unk_score = -tf.math.log(1 / unk_score - 1)  # [B]
      log_unk_score = log_unk_score[:, None]  # [B, 1]
    elif self._unknown_logits == "radii":
      log_unk_score = (min_dist - beta_) / gamma_  # [B]
      log_unk_score = tf.expand_dims(log_unk_score, -1)

    # log_unk_score = (min_dist - beta_) / gamma_  # [B]
    # log_unk_score = tf.expand_dims(log_unk_score, -1)

    # Mask out the unused bits.
    k_idx = tf.cast(tf.range(K + 1), dtype)  # [B, K+1]
    k_idx = tf.tile(tf.expand_dims(k_idx, 0), [x.shape[0], 1])  # [B, K+1]
    # cur_num = tf.reduce_sum(
    #     tf.cast(tf.greater(count, 0), dtype), [1], keepdims=True)  # [B, 1]

    # tf.print('cur_num', cur_num[7])
    # tf.print('count', count[7], summarize=100)
    valid = tf.less(k_idx, cur_num)  # [B, K+1]
    logits = tf.where(valid, logits, -LOGINF)
    if add_new:
      if self._fix_unknown:
        addition = tf.equal(k_idx, self._unknown_id)  # [B, K+1]
      else:
        addition = tf.equal(k_idx, cur_num)  # [B, k+1]
      logits = tf.where(addition, log_unk_score, logits)
    return logits

  def retrieve_all(self, x, storage, count, t, add_new=True):
    """See ProtoMemory for documentation."""
    raise NotImplementedError()
