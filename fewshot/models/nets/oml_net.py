"""Online meta-learning.
Following Javed, K., White, Martha. Meta-Learning Representations for Continual
Learning.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.registry import RegisterModel
from fewshot.models.nets.episode_recurrent_net import EpisodeRecurrentNet
from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA


@RegisterModel("oml_net")
class OMLNet(EpisodeRecurrentNet):

  def __init__(self, config, backbone, memory, dtype=tf.float32):
    super(OMLNet, self).__init__(config, backbone, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory

  def mask(self, y, k, K):
    """Mask out non-possible bits."""
    LOGINF = 1e5
    if self.config.fix_unknown:
      # The last dimension is always useful.
      k = tf.cast(k, y.dtype)
      mask = tf.greater(tf.range(K, dtype=y.dtype),
                        tf.expand_dims(k, 1))  # [B, NO]
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
    # Current seen maximum classes.
    k = tf.zeros([B], dtype=y.dtype) - 1
    states = self.memory.get_initial_state(B)

    for t in tf.range(T):
      x_ = self.slice_time(x, t)  # [B, D]
      y_ = self.slice_time(y, t)  # [B]

      y_cls_, y_unk_, states = self.memory(x_, y_, *states)
      y_unk_ = tf.math.sigmoid(y_unk_)  # [B, 1]
      y_cls_ = self.mask(y_cls_, k, K)  # [B, NO]
      y_cls_ = tf.nn.softmax(y_cls_)  # [B, NO]
      y_pred = y_pred.write(t, tf.concat([y_cls_, y_unk_[:, None]], axis=-1))
      k = tf.maximum(k, y_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    assert x_test is None
    return y_pred

  def compute_loss(self, logits, labels):
    # Cross entropy loss, now not using logits.
    labels_onehot = tf.one_hot(labels, tf.shape(logits)[-1])  # [B, T, K]
    xent = -tf.math.log(tf.reduce_sum(logits * labels_onehot, [-1]))  # [B, T]
    return xent

  def predict_id(self, logits):
    return tf.argmax(logits, axis=-1)

  @property
  def memory(self):
    return self._memory


@RegisterModel("oml_sigmoid_net")
class OMLSigmoidNet(EpisodeRecurrentSigmoidNet):

  def __init__(self, config, backbone, memory, dtype=tf.float32):
    super(OMLSigmoidNet, self).__init__(config, backbone, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory

  def mask(self, y, k, K):
    """Mask out non-possible bits."""
    LOGINF = 1e5
    if self.config.fix_unknown:
      # The last dimension is always useful.
      k = tf.cast(k, y.dtype)
      mask = tf.greater(tf.range(K, dtype=y.dtype),
                        tf.expand_dims(k, 1))  # [B, NO]
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
    # Current seen maximum classes.
    k = tf.zeros([B], dtype=y.dtype) - 1
    states = self.memory.get_initial_state(B)

    for t in tf.range(T):
      x_ = self.slice_time(x, t)  # [B, D]
      y_ = self.slice_time(y, t)  # [B]
      y_cls_, y_unk_, states = self.memory(x_, y_, *states)
      y_cls_ = self.mask(y_cls_, k, K)  # [B, NO]
      y_pred = y_pred.write(t, tf.concat([y_cls_, y_unk_[:, None]], axis=-1))
      k = tf.maximum(k, y_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    assert x_test is None
    return y_pred

  @property
  def memory(self):
    return self._memory
