"""LSTM based networks with extra sigmoid output for unknown.

Note: Here everything is the same as LSTMNet, except it inherits from
EpisodeRecurrentSigmoidNet.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.modules.nnlib import Linear
from fewshot.models.registry import RegisterModel

LOGINF = 1e5


@RegisterModel("lstm_sigmoid_net")
class LSTMSigmoidNet(EpisodeRecurrentSigmoidNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(LSTMSigmoidNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory
    self._nclassout = config.num_classes + 1
    self._readout_layer = Linear('readout', memory.nout, self._nclassout)

  def readout(self, m, *h):
    """Read out from memory.

    Args:
      m: [B, D]: Memory read head content.
      h: Hidden states.

    Returns:
      c: [B, K]: Classification logits.
    """
    return self._readout_layer(m)

  def run_memory(self, x, y_prev, *h):
    """Run forward pass on the memory.

    Args:
      x: [B, D]. Input features.
      y_prev: [B, K]. Class ID from previous time step.

    Returns:
      m: [B, D]. Memory read out features.
      h: List. Memory states.
    """
    x_and_y = tf.concat([x, y_prev], axis=-1)  # [B, D+NO]
    return self.memory(x_and_y, *h)

  def mask(self, y, k, K):
    """Mask out non-possible bits."""
    if self.config.fix_unknown:
      # The last dimension is always useful.
      k = tf.cast(k, y.dtype)
      mask = tf.greater(
          tf.concat(
              [tf.range(K, dtype=y.dtype),
               tf.constant([-1], dtype=y.dtype)],
              axis=0), tf.expand_dims(k, 1))  # [B, NO]
    else:
      mask = tf.greater(tf.range(K, dtype=y.dtype),
                        tf.expand_dims(k + 1, 1))  # [B, NO]
    mask.set_shape([y.shape[0], y.shape[1]])
    y = tf.where(mask, -LOGINF, y)
    return y

  def forward(self, x, y, s=None, x_test=None, is_training=tf.constant(True)):
    """Make a forward pass.

    Args:
      x: [B, T, ...]. Support examples.
      y: [B, T, ...]. Support examples labels.
      x_test: [B, T', ...]. Query examples.
      is_training. Bool. Whether in training mode.

    Returns:
      y_pred: [B, T]. Support example prediction.
      y_pred_test: [B, T']. Query example prediction, if exists.
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
    h_ = self.memory.get_initial_state(B)
    m_ = tf.zeros([B, self.memory.nout], dtype=self.dtype)
    mask = tf.zeros([B, Nout], dtype=tf.bool)  # [B, NO]

    if s is not None:
      y_prev_10 = tf.zeros([B, Nout * 2], self.dtype)  # [B, NO]
    else:
      y_prev_10 = tf.zeros([B, Nout], self.dtype)  # [B, NO]

    for t in tf.range(T):
      x_ = self.slice_time(x, t)  # [B, D]
      if t > 0:
        y_prev = self.slice_time(y, t - 1)  # [B]
        y_prev_10 = tf.one_hot(y_prev, Nout)  # [B, NO]
        y_prev_10.set_shape([x.shape[0], Nout])

        # Add stage info.
        if s is not None:
          s_ = self.slice_time(s, t)  # [B]
          s_10 = tf.one_hot(s_, Nout)
          s_10.set_shape([x.shape[0], Nout])
          y_prev_10 = tf.concat([y_prev_10, s_10], axis=1)
        k = tf.maximum(k, y_prev)

      m_, h_ = self.run_memory(x_, y_prev_10, *h_)
      y_pred_ = self.readout(m_, *h_)  # [B, NO]
      y_pred_ = self.mask(y_pred_, k, K)  # [B, NO]
      y_pred = y_pred.write(t, y_pred_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])

    # Query set.
    if x_test is not None:
      x_test = self.run_backbone(x_test, is_training=is_training)  # [B, M, D]
      y_pred_test = []
      T_test = tf.constant(x_test.shape[1])
      D = tf.constant(x_test.shape[-1])
      D2 = tf.constant(m_.shape[-1])
      M = tf.constant(x_test.shape[1])
      x_test_ = tf.reshape(x_test, [-1, D])  # [BM, D]
      y_prev_10_ = tf.reshape(
          tf.tile(tf.expand_dims(y_prev_10, 1), [1, M, 1]),
          [-1, Nout])  # [BM, NO]
      h_ = self.memory.expand_state(M, *h_)
      m_, _ = self.run_memory(x_test_, y_prev_10_, *h_)
      y_pred_ = self.readout(m_, *h_)  # [BM, NO]
      y_pred_ = tf.reshape(y_pred_, [-1, M, Nout])
      mask = tf.expand_dims(mask, 1)
      y_pred_test = tf.where(mask, -LOGINF, y_pred_)
      return y_pred, y_pred_test
    else:
      return y_pred

  @property
  def memory(self):
    """Memory module"""
    return self._memory
