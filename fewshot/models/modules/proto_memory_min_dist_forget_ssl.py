"""A semi-supervised prototypical memory module with minimum distance.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.proto_memory_min_dist_ssl import SemiSupervisedMinDistProtoMemory  # NOQA
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.lstm import LSTM
from fewshot.models.modules.nnlib import Linear
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()

LOGINF = 1e6


@RegisterModule('ssl_min_dist_forget_proto_memory')
class SemiSupervisedMinDistForgetProtoMemory(SemiSupervisedMinDistProtoMemory):

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
    assert False, 'hey3'
    super(SemiSupervisedMinDistProtoMemory, self).__init__(
        name,
        dim,
        max_classes=max_classes,
        fix_unknown=fix_unknown,
        unknown_id=unknown_id,
        similarity=similarity,
        dtype=dtype)

    self._controller_type = 'linear'
    # self._controller_type = 'lstm'
    self._radius_init = radius_init
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

    with variable_scope(name):
      if self._controller_type == 'lstm':
        self._ctrl_lstm = LSTM(
            "ctrl_lstm", dim, dim, layernorm=False, dtype=dtype)
        self._ctrl_readout = Linear(
            "ctrl_readout",
            dim,
            1,
            w_init=lambda: tf.ones([dim, 1]),
            b_init=lambda: tf.zeros([1]))
      elif self._controller_type == 'linear':
        self._ctrl_readout = Linear(
            "ctrl_readout",
            dim,
            1,
            # w_init=lambda: self._get_normal_init([dim, 1])() * 0.001,
            w_init=lambda: tf.ones([dim, 1]) * 0.001,
            b_init=lambda: tf.zeros([1]))

  def adjust_count(self, x, y, y_soft, storage, count):
    bidx = tf.range(tf.shape(y)[0])  # [B]
    y2_idx = tf.argmax(y_soft[:, :-1], axis=-1, output_type=y.dtype)
    unk = tf.constant(self._unknown_id, dtype=y.dtype)
    y_unk2 = tf.math.sigmoid(y_soft[:, -1:])
    y_forget = tf.where(
        tf.greater(y_unk2[:, 0], 0.5), unk,
        tf.where(tf.equal(y, unk), y, y2_idx))  # [B]
    y_forget = tf.cast(y_forget, bidx.dtype)  # [B]
    idx = tf.stack([bidx, y_forget], axis=1)  # [B, 2]

    # Compute the difference between cluster center and the new input.
    curr_val = tf.gather_nd(storage, idx)  # [B, D]
    storage_old = tf.scatter_nd(idx, curr_val, storage.shape)
    storage_new = tf.scatter_nd(idx, x, storage.shape)
    delta = storage_old - storage_new

    if self._controller_type == 'lstm':
      # Forgetting on the size of the cluster.
      ctrl_inp = delta  # [B, M, D] -> [BM, D]
      ctrl_inp = tf.reshape(ctrl_inp, [-1, ctrl_inp.shape[-1]])
      ctrl_out, (ctrl_c_new, ctrl_h_new) = self._ctrl_lstm(
          ctrl_inp, ctrl_c, ctrl_h)  # [BM, D]
      # Bias forget gate with 1.0
      f = self._ctrl_readout(ctrl_out) + 1.0  # [BM, 1]
      f = tf.reshape(f, [x.shape[0], -1])  # [B, M]
      count = count * f  # [B, M]
    elif self._controller_type == 'linear':
      ctrl_inp = delta  # [B, M, D] -> [BM, D]
      ctrl_inp = tf.reshape(ctrl_inp, [-1, ctrl_inp.shape[-1]])
      # Bias forget gate with 1.0
      f = self._ctrl_readout(ctrl_inp) + 1.0  # [BM, 1]
      f = tf.reshape(f, [x.shape[0], -1])  # [B, M]

      # Investigate whether we need to initialize it smaller, or
      # need to have exceeding the range.
      # f = tf.minimum(tf.maximum(f, 0.0), 1.0)  # [B, M]
      # f = tf.maximum(f, 0.0)

      tf.print('f1 min', tf.reduce_min(f), 'max', tf.reduce_max(f), 'avg',
               tf.reduce_mean(f))

      tf.print('y_forget', y_forget)

      f = tf.where(tf.equal(y_forget[:, None], self.unknown_id), 1.0,
                   f)  # [B, M]

      tf.print('f min', tf.reduce_min(f), 'max', tf.reduce_max(f), 'avg',
               tf.reduce_mean(f))
      # tf.summary.scalar('proto forget/min', tf.reduce_min(f))
      # tf.summary.scalar('proto forget/max', tf.reduce_max(f))
      # tf.summary.scalar('proto forget/mean', tf.reduce_mean(f))
      count = count * f  # [B, M]
    return count

  def store(self, x, y, storage, count, y_soft=None):
    """Store a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
      storage: Storage. [B, K, D].
      count: Count. [B, K].
      y_soft: Soft label. [B, K].
    """
    if y_soft is not None:
      count = self.adjust_count(x, y, y_soft, storage, count)

    return super(SemiSupervisedMinDistForgetProtoMemory, self).store(
        x, y, storage, count, y_soft=y_soft)

  def forward_one(self,
                  x,
                  y,
                  t,
                  *states,
                  add_new=tf.constant(True),
                  is_training=tf.constant(True),
                  ssl_store=tf.constant(True),
                  **kwargs):

    if self._controller_type == 'lstm':
      storage, count, ctrl_c, ctrl_h = states
    else:
      storage, count = states
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

    # Store with the new count.
    proto_states_ssl = self.store(x, y, storage, count, y_soft=y_soft)
    proto_states_nossl = self.store(x, y, storage, count)
    storage_new = tf.where(ssl_store[:, None, None], proto_states_ssl[0],
                           proto_states_nossl[0])
    count_new = tf.where(ssl_store[:, None], proto_states_ssl[1],
                         proto_states_nossl[1])

    storage_new.set_shape([x.shape[0], self.max_classes + 1, self._dim])
    count_new.set_shape([x.shape[0], self.max_classes + 1])

    if self._controller_type == 'lstm':
      return y_, (storage_new, count_new, ctrl_c_new, ctrl_h_new)
    elif self._controller_type == 'linear':
      return y_, (storage_new, count_new)

  def get_initial_state(self, bsize):
    """Get initial states."""
    storage, count = super(SemiSupervisedMinDistForgetProtoMemory,
                           self).get_initial_state(bsize)

    if self._controller_type == 'lstm':
      ctrl_c, ctrl_h = self._ctrl_lstm.get_initial_state(
          bsize * storage.shape[1])
      return storage, count, ctrl_c, ctrl_h
    else:
      return storage, count
