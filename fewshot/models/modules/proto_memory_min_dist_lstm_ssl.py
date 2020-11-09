"""A semi-supervised prototypical memory module with minimum distance.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.proto_memory_min_dist_ssl import SemiSupervisedMinDistProtoMemory  # NOQA
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.gru import LSTM1DMod
# from fewshot.models.modules.nnlib import Linear
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()

LOGINF = 1e6


@RegisterModule('ssl_min_dist_lstm_proto_memory')
class SemiSupervisedMinDistLSTMProtoMemory(SemiSupervisedMinDistProtoMemory):

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
               dtype=tf.float32):
    super(SemiSupervisedMinDistProtoMemory, self).__init__(
        name,
        dim,
        max_classes=max_classes,
        fix_unknown=fix_unknown,
        unknown_id=unknown_id,
        similarity=similarity,
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
      self._storage = LSTM1DMod(
          "storage",
          dim,
          dim,  # layernorm=False, 
          dtype=dtype)

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

  def retrieve(self,
               x,
               h,
               count,
               t,
               beta=None,
               gamma=None,
               temp=None,
               add_new=tf.constant(True),
               is_training=tf.constant(True)):
    """See ProtoMemory for documentation."""
    storage, _ = tf.split(h, 2, axis=-1)  # [B, K, D]
    return super(SemiSupervisedMinDistLSTMProtoMemory, self).retrieve(
        x,
        storage,
        count,
        t,
        beta=beta,
        gamma=gamma,
        temp=temp,
        add_new=add_new,
        is_training=is_training)

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
    # Change y_soft to normal value
    h_new_ssl, count_new_ssl = self.store(
        x, y, h_last, count_last, y_soft=None)
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
    # assert y_soft is None, 'Not supported'
    y_soft = None
    B = tf.shape(y)[0]
    K = tf.shape(h_last)[1]
    D = self.dim
    bidx = tf.range(B)  # [B]
    y = tf.cast(y, bidx.dtype)  # [B]
    idx = tf.stack([bidx, y], axis=1)  # [B, 2]
    # tf.print('B', B, 'K', K, 'D', D)
    # print(h_last.shape, B, K, D)
    h_last = tf.reshape(h_last, [B * K, -1])  # [BK, D]
    curr_cnt = tf.expand_dims(tf.gather_nd(count, idx), 1)  # [B, 1]
    c_last_, h_last_ = tf.split(h_last, 2, axis=-1)  # [BK, D]

    # Do not increment for unknowns.
    inc = 1.0 - tf.cast(tf.equal(y, self._unknown_id), self.dtype)  # [B]
    inp = tf.scatter_nd(idx, x, [B, K, D])  # [B, K, D]
    inp = tf.reshape(inp, [B * K, D])

    # TODO disable mask for y == unk_id
    mask = tf.scatter_nd(idx, tf.ones([B, 1]), [B, K, 1])  # [B, K, D]
    mask = tf.reshape(mask, [B * K, 1])
    mask = mask * tf.reshape(tf.tile(inc[:, None, None], [1, K, 1]), [-1, 1])

    _, (c_new, h_new) = self.storage(inp, c_last_, h_last_)  # [BK, D]
    h_new = tf.concat([c_new, h_new], axis=-1)  # [BK, 2D]
    h_new = tf.where(tf.greater(mask, 0.5), h_new, h_last)  # [BK, 2D]
    h_new = tf.reshape(h_new, [B, K, -1])  # [B, K, 2D]

    # TODO maybe consider use f_gate to decay count.
    # count = count * tf.scatter_nd(idx, f_gate * inc + (1.0 - inc), count.shape)
    count_new = count + tf.scatter_nd(idx, inc, count.shape)  # [B]

    # if y_soft is not None:
    #   mask = tf.cast(tf.greater(count, 0), y_soft.dtype)  # [B, K]
    #   y_soft_mask = y_soft * mask  # [B, K]

    #   # TODO maybe consider doing argmax here for better performance.
    #   unk_mask = tf.cast(tf.equal(y, self._unknown_id), y_soft.dtype)  # [B]
    #   unk_mask = tf.reshape(unk_mask, [-1, 1, 1])  # [B, 1, 1]

    #   # [B, 1, D] * [B, K, 1] * [B, 1, 1] = [B, K, D]
    #   unk_update = tf.expand_dims(x, 1) * tf.expand_dims(y_soft_mask,
    #                                                      -1) * unk_mask
    #   storage_new = storage_new * tf.expand_dims(count_new, -1) + unk_update
    #   count_new += y_soft_mask * unk_mask[:, :, 0]
    #   storage_new /= tf.expand_dims(
    #       count_new + tf.cast(tf.equal(count_new, 0), self.dtype), -1)
    return h_new, count_new

  def get_initial_state(self, bsize):
    """Get initial states."""
    K = self.max_classes
    if self._fix_unknown:
      K += 1
    c, h = self.storage.get_initial_state(bsize * K)
    c = tf.reshape(c, [bsize, K, self.dim])
    h = tf.reshape(h, [bsize, K, self.dim])
    ch = tf.concat([c, h], axis=-1)
    count = tf.zeros([bsize, K], dtype=self.dtype)
    return ch, count

  @property
  def storage(self):
    return self._storage
