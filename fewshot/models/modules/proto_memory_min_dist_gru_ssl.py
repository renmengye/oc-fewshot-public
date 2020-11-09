"""A semi-supervised prototypical memory module with minimum distance.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.proto_memory_min_dist_ssl import SemiSupervisedMinDistProtoMemory  # NOQA
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.gru import GRU1DMod
# from fewshot.models.modules.nnlib import Linear
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()

LOGINF = 1e6


@RegisterModule('ssl_min_dist_gru_proto_memory')
class SemiSupervisedMinDistGRUProtoMemory(SemiSupervisedMinDistProtoMemory):

  def __init__(self,
               name,
               dim,
               radius_init,
               max_classes=20,
               fix_unknown=False,
               unknown_id=None,
               similarity="euclidean",
               static_beta_gamma=True,
               unknown_logits="radii",
               radius_init_write=None,
               use_ssl_beta_gamma_write=True,
               temp_init=10.0,
               dtype=tf.float32):
    assert unknown_logits == 'radii'
    super(SemiSupervisedMinDistGRUProtoMemory, self).__init__(
        name,
        dim,
        radius_init,
        max_classes=max_classes,
        fix_unknown=fix_unknown,
        unknown_id=unknown_id,
        similarity=similarity,
        unknown_logits=unknown_logits,
        temp_init=temp_init,
        dtype=dtype)
    self._radius_init = radius_init
    log.info('Radius init {}'.format(radius_init))
    if radius_init_write is not None:
      self._radius_init_write = radius_init_write
      log.info('Radius init write {}'.format(radius_init_write))
    else:
      self._radius_init_write = radius_init
    self._use_ssl_beta_gamma_write = use_ssl_beta_gamma_write

    with variable_scope(name):
      self._storage = GRU1DMod(
          "storage", dim, dim, layernorm=False, dtype=dtype)

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
                  h_last,
                  count_last,
                  add_new=tf.constant(True),
                  is_training=tf.constant(True),
                  ssl_store=tf.constant(True),
                  **kwargs):
    y_ = self.retrieve(
        x, h_last, count_last, t, add_new=add_new,
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
        h_last,
        count_last,
        t,
        beta=beta2,
        gamma=gamma2,
        add_new=add_new,
        is_training=is_training)  # [B, K]

    y_unk2 = tf.math.sigmoid(y2_[:, -1:])
    y_soft = tf.concat([tf.nn.softmax(y2_[:, :-1]) * (1.0 - y_unk2), y_unk2],
                       axis=-1)

    # Store with the new count.
    # TODO: Change y_soft to normal value
    h_new_ssl, count_new_ssl = self.store(x, y, h_last, count_last, y_soft)
    # h_new_ssl, count_new_ssl = self.store(
    #     x, y, h_last, count_last, y_soft=None)
    h_new_nossl, count_new_nossl = self.store(x, y, h_last, count_last)
    h_new = tf.where(ssl_store[:, None, None], h_new_ssl, h_new_nossl)
    count_new = tf.where(ssl_store[:, None], count_new_ssl, count_new_nossl)
    return y_, h_new, count_new

  def store(self, x, y, h_last, count, y_soft=None):
    """Store a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
    """
    # assert y_soft is not None, 'Not supported'
    # y_soft = None
    B = tf.shape(y)[0]
    K = tf.shape(h_last)[1]
    D = self.dim
    bidx = tf.range(B)  # [B]
    y = tf.cast(y, bidx.dtype)  # [B]
    idx = tf.stack([bidx, y], axis=1)  # [B, 2]
    h_last = tf.reshape(h_last, [B * K, D])  # [BK, D]
    curr_cnt = tf.expand_dims(tf.gather_nd(count, idx), 1)  # [B, 1]

    # Do not increment for unknowns.
    inc = 1.0 - tf.cast(tf.equal(y, self._unknown_id), self.dtype)  # [B]
    inp = tf.scatter_nd(idx, x, [B, K, D])  # [B, K, D]
    inp = inp * inc[:, None, None]  # set last storage dim to zero.

    # # TODO disable mask for y == unk_id
    # mask = tf.scatter_nd(idx, tf.ones([B, D]), [B, K, D])  # [B, K, D]
    # mask = tf.reshape(mask, [B * K, D])
    # mask = mask * tf.reshape(tf.tile(inc[:, None, None], [1, K, 1]), [-1, 1])
    # tf.print('y soft3', y_soft)

    if y_soft is not None:
      use_mask = tf.cast(tf.greater(count, 0), y_soft.dtype)  # [B, K]
      # y_soft = tf.concat([y_soft[:, :-1], tf.zeros([B, 1])], axis=1)  # [B, K]
      y_soft_mask = y_soft * use_mask  # [B, K]
      unk_mask = tf.cast(tf.equal(y, self._unknown_id), y_soft.dtype)  # [B]
      unk_mask = tf.reshape(unk_mask, [-1, 1, 1])  # [B, 1, 1]

      # -------------------------------------------
      # V1: normal version multiply soft mask.
      # [B, 1, D] * [B, K, 1] * [B, 1, 1] = [B, K, D]
      inp += tf.expand_dims(x, 1) * tf.expand_dims(y_soft_mask, -1) * unk_mask

      # -------------------------------------------
      # V2: argmax version, doesn't multiply soft mask.
      # tf.print('y_soft', y_soft)
      # y_soft_idx = tf.cast(tf.argmax(y_soft, axis=-1), bidx.dtype)  # [B]
      # idx_unk = tf.stack([bidx, y_soft_idx], axis=1)  # [B, 2]
      # inp_unk = tf.scatter_nd(idx_unk, x, [B, K, D])  # [B, K, D]
      # inp += inp_unk * unk_mask

    inp = tf.reshape(inp, [B * K, D])
    h_new, _ = self.storage(inp, h_last)  # [BK, D]
    # Sparse update.
    # h_new = tf.where(tf.greater(mask, 0.5), h_new, h_last)
    h_new = tf.reshape(h_new, [B, K, D])  # [B, K, D]

    # TODO maybe consider use f_gate to decay count.
    count_new = count + tf.scatter_nd(idx, inc, count.shape)  # [B]
    return h_new, count_new

  def get_initial_state(self, bsize):
    """Get initial states."""
    K = self.max_classes
    if self._fix_unknown:
      K += 1
    h = self.storage.get_initial_state(bsize * K)
    h = tf.reshape(h, [bsize, K, self.dim])
    count = tf.zeros([bsize, K], dtype=self.dtype)
    return h, count

  @property
  def storage(self):
    return self._storage
