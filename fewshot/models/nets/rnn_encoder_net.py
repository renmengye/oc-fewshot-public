"""RNN encoder for analyzing the intermediate variables.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("rnn_encoder_net")
class RNNEncoderNet(EpisodeRecurrentSigmoidNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(RNNEncoderNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory
    self._nclassout = config.num_classes + 1

  def forward(self, x, y, *states, is_training=tf.constant(True), **kwargs):
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
    K = tf.constant(self.config.num_classes)
    out = tf.TensorArray(self.dtype, size=T)
    beta = tf.TensorArray(self.dtype, size=T)
    beta2 = tf.TensorArray(self.dtype, size=T)
    gamma = tf.TensorArray(self.dtype, size=T)
    gamma2 = tf.TensorArray(self.dtype, size=T)
    count = tf.TensorArray(self.dtype, size=T)
    Nout = self._nclassout
    ssl_store = tf.ones([B, T], dtype=tf.bool)
    # Current seen maximum classes.
    if len(states) == 0:
      h_ = self.memory.get_initial_state(B, tf.shape(x)[-1])
    else:
      h_ = states

    for t in tf.range(T):
      x_ = self.slice_time(x, t)  # [B, D]
      y_ = self.slice_time(y, t)  # [B]
      out_, (beta_, gamma_, beta2_, gamma2_, count_), h_ = self.memory(
          t, x_, y_, *h_, store=tf.greater(t, 0), ssl_store=ssl_store[:, t])
      out = out.write(t, out_)
      beta = beta.write(t, beta_)
      beta2 = beta2.write(t, beta2_)
      gamma = gamma.write(t, gamma_)
      gamma2 = gamma2.write(t, gamma2_)
      count = count.write(t, count_)

    out = tf.transpose(out.stack(), [1, 0, 2])  # [B, T, D]
    beta = tf.transpose(beta.stack(), [1, 0])  # [B, T]
    gamma = tf.transpose(gamma.stack(), [1, 0])  # [B, T]
    beta2 = tf.transpose(beta2.stack(), [1, 0])  # [B, T]
    gamma2 = tf.transpose(gamma2.stack(), [1, 0])  # [B, T]
    count = tf.transpose(count.stack(), [1, 0])  # [B, T]
    return out, (beta, gamma, beta2, gamma2, count)

  @property
  def memory(self):
    """Memory module"""
    return self._memory
