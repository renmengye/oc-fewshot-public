"""Contextual prototypical memory network

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.nets.episode_recurrent_sigmoid_trunc_net import EpisodeRecurrentSigmoidTruncNet  # NOQA
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("cpm_net")
@RegisterModel("proto_plus_rnn_net")
class CPMNet(EpisodeRecurrentSigmoidNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(CPMNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory
    self._nclassout = config.num_classes + 1

  def run_memory(self,
                 t,
                 x,
                 y_prev,
                 *h,
                 store=True,
                 ssl_store=tf.constant(True)):
    """Run forward pass on the memory.

    Args:
      x: [B, D]. Input features.
      y_prev: [B, K]. Class ID from previous time step.

    Returns:
      m: [B, D]. Memory read out features.
      h: List. Memory states.
    """
    return self.memory(t, x, y_prev, *h, store=store, ssl_store=ssl_store)

  def forward(self, x, y, *states, is_training=tf.constant(True), **kwargs):
    """Make a forward pass.

    Args:
      x: [B, T, ...]. Support examples.
      y: [B, T, ...]. Support examples labels.
      is_training. Bool. Whether in training mode.

    Returns:
      y_pred: [B, T]. Support example prediction.
    """
    x = self.run_backbone(x, is_training=is_training)
    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    y_pred = tf.TensorArray(self.dtype, size=T)
    # Maximum total number of classes.
    K = tf.constant(self.config.num_classes)
    Nout = self._nclassout
    # Current seen maximum classes.
    k = tf.zeros([B], dtype=y.dtype) - 1
    if len(states) == 0:
      h_ = self.memory.get_initial_state(B)
    else:
      h_ = states

    mask = tf.zeros([B, Nout], dtype=tf.bool)  # [B, NO]

    if self.config.ssl_store_schedule:
      log.info("Using probabilistic semisupervised store schedule")
      store_prob = tf.compat.v1.train.piecewise_constant(
          self._step, [2000, 4000, 6000, 8000, 10000],
          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
      ssl_store = tf.less(tf.random.uniform([B, T], 0.0, 1.0), store_prob)
      self._ssl_store = ssl_store
    else:
      ssl_store = tf.ones([B, T], dtype=tf.bool)

    for t in tf.range(T):
      x_ = self.slice_time(x, t)  # [B, D]
      # tf.print('t inp', t, tf.reduce_mean(x_), tf.reduce_max(x_),
      #          tf.reduce_min(x_))
      y_ = self.slice_time(y, t)  # [B]
      y_pred_, y_unk_, rnnout, h_ = self.run_memory(
          t, x_, y_, *h_, store=tf.greater(t, 0), ssl_store=ssl_store[:, t])
      y_pred = y_pred.write(
          t, tf.concat([y_pred_[:, :-1], y_unk_[:, None]], axis=-1))

    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])

    if len(states) == 0:
      return y_pred
    else:
      return y_pred, h_

  @property
  def memory(self):
    """Memory module"""
    return self._memory

  @tf.function
  def eval_step(self, x, y, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.

    Returns:
      logits: [B, T, Kmax], prediction.
    """
    logits = self.forward(x, y, is_training=tf.constant(False))
    return logits


@RegisterModel("cpm_trunc_net")
@RegisterModel("proto_plus_rnn_trunc_net")
class CPMTruncNet(EpisodeRecurrentSigmoidTruncNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(CPMTruncNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory
    self._nclassout = config.num_classes + 1

  def run_memory(self, t, x, y_prev, *h, store=True):
    """Run forward pass on the memory.

    Args:
      x: [B, D]. Input features.
      y_prev: [B, K]. Class ID from previous time step.

    Returns:
      m: [B, D]. Memory read out features.
      h: List. Memory states.
    """
    return self.memory(t, x, y_prev, *h, store=store)

  def forward(self,
              x,
              y,
              t0,
              dt,
              *states,
              is_training=tf.constant(True),
              **kwargs):
    """Make a forward pass.

    Args:
      x: [B, T, ...]. Support examples.
      y: [B, T, ...]. Support examples labels.
      is_training. Bool. Whether in training mode.

    Returns:
      y_pred: [B, T]. Support example prediction.
    """
    x = self.run_backbone(x, is_training=is_training)
    B = tf.shape(x)[0]
    T = dt
    y_pred = tf.TensorArray(self.dtype, size=T)
    # Maximum total number of classes.
    K = tf.constant(self.config.num_classes)
    Nout = self._nclassout
    # Current seen maximum classes.
    if len(states) == 0:
      h_ = self.memory.get_initial_state(B)
    else:
      h_ = states

    # Need one more round to store the item.
    for t in tf.range(T + 1):
      if tf.less(t, T):
        x_ = self.slice_time(x, t)  # [B, D]
        y_ = self.slice_time(y, t)
      else:
        x_ = self.slice_time(x, T - 1)
        y_ = self.slice_time(y, T - 1)

      y_pred_, y_unk_, _, h_ = self.run_memory(
          t, x_, y_, *h_, store=tf.greater(t0 + t, 0))

      if tf.less(t, T):
        y_pred = y_pred.write(
            t, tf.concat([y_pred_[:, :-1], y_unk_[:, None]], axis=-1))
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    if len(states) == 0:
      return y_pred
    else:
      return y_pred, h_

  @property
  def memory(self):
    """Memory module"""
    return self._memory
