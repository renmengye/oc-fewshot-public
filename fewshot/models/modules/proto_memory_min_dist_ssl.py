"""A semi-supervised prototypical memory module with minimum distance.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.proto_memory_ssl import SemiSupervisedProtoMemory  # NOQA
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()

LOGINF = 1e6


@RegisterModule('ssl_min_dist_proto_memory')
class SemiSupervisedMinDistProtoMemory(SemiSupervisedProtoMemory):

  def __init__(self,
               name,
               dim,
               radius_init,
               max_classes=20,
               fix_unknown=False,
               unknown_id=None,
               similarity="euclidean",
               static_beta_gamma=True,
               radius_init_write=None,
               use_ssl_beta_gamma_write=True,
               unknown_logits="radii",
               temp_init=10.0,
               dtype=tf.float32):
    super(SemiSupervisedMinDistProtoMemory, self).__init__(
        name,
        dim,
        max_classes=max_classes,
        fix_unknown=fix_unknown,
        unknown_id=unknown_id,
        similarity=similarity,
        temp_init=temp_init,
        dtype=dtype)
    self._radius_init = radius_init
    self._unknown_logits = unknown_logits
    log.info('Radius init {}'.format(radius_init))
    if radius_init_write is not None:
      self._radius_init_write = radius_init_write
      log.info('Radius init write {}'.format(radius_init_write))
    else:
      self._radius_init_write = radius_init
    self._use_ssl_beta_gamma_write = use_ssl_beta_gamma_write
    if static_beta_gamma:
      with variable_scope(name):
        self._beta = self._get_variable(
            "beta", self._get_constant_init([], radius_init))
        self._gamma = self._get_variable("gamma",
                                         self._get_constant_init([], 1.0))

        self._beta2 = self._get_variable(
            "beta2", self._get_constant_init([], self._radius_init_write))
        self._gamma2 = self._get_variable("gamma2",
                                          self._get_constant_init([], 1.0))

  def forward_one(self,
                  x,
                  y,
                  t,
                  storage,
                  count,
                  add_new=tf.constant(True),
                  is_training=tf.constant(True),
                  ssl_store=tf.constant(True),
                  **kwargs):
    y_ = self.retrieve(
        x, storage, count, t, add_new=add_new,
        is_training=is_training)  # [B, K]

    if self._use_ssl_beta_gamma_write:
      beta2 = self._beta2
      gamma2 = self._gamma2
    else:
      log.info('Not using separate beta gamma for SSL')
      beta2 = None
      gamma2 = None
    y2_ = self.retrieve(
        x,
        storage,
        count,
        t,
        beta=beta2,
        gamma=gamma2,
        add_new=add_new,
        is_training=is_training)  # [B, K]

    y_unk2 = tf.math.sigmoid(y2_[:, -1:])
    y_soft = tf.concat([tf.nn.softmax(y2_[:, :-1]) * (1.0 - y_unk2), y_unk2],
                       axis=-1)

    proto_states_ssl = self.store(x, y, storage, count, y_soft=y_soft)
    proto_states_nossl = self.store(x, y, storage, count)
    storage_new = tf.where(ssl_store[:, None, None], proto_states_ssl[0],
                           proto_states_nossl[0])
    count_new = tf.where(ssl_store[:, None], proto_states_ssl[1],
                         proto_states_nossl[1])
    storage_new.set_shape([x.shape[0], self.max_classes + 1, self._dim])
    count_new.set_shape([x.shape[0], self.max_classes + 1])
    return y_, (storage_new, count_new)

  def retrieve(self,
               x,
               storage,
               count,
               t,
               beta=None,
               gamma=None,
               temp=None,
               add_new=tf.constant(True),
               is_training=tf.constant(True)):
    """See ProtoMemory for documentation."""
    x = tf.expand_dims(x, 1)  # [B, 1, D]
    prototypes = storage
    K = self.max_classes
    dtype = self.dtype
    beta_ = beta if beta is not None else self._beta
    gamma_ = gamma if gamma is not None else self._gamma

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

    if temp is not None:
      logits *= temp  # Additonal temperature.

    k_idx = tf.cast(tf.range(K + 1), dtype)  # [B, K+1]
    k_idx = tf.tile(tf.expand_dims(k_idx, 0), [x.shape[0], 1])  # [B, K+1]
    valid = tf.less(k_idx, cur_num)  # [B, K+1]
    logits = tf.where(valid, logits, -LOGINF)
    if add_new:
      if self._fix_unknown:
        addition = tf.equal(k_idx, self._unknown_id)  # [B, K+1]
      else:
        addition = tf.equal(k_idx, cur_num)  # [B, k+1]
      logits = tf.where(addition, log_unk_score, logits)
    return logits
