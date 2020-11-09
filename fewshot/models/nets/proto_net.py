"""A prototypical network.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nets.net import Net
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("proto_net")
class ProtoNet(Net):

  def __init__(self, config, backbone, dtype=tf.float32):
    super(ProtoNet, self).__init__()
    self._backbone = backbone
    self._config = config
    assert self.config.num_classes > 0, 'Must specify number of output classes'
    opt_config = self.config.optimizer_config
    gs = tf.Variable(0, dtype=tf.int64, name='step', trainable=False)
    self._step = gs
    self._wd = backbone.config.weight_decay
    self._var_to_optimize = None
    self._regularized_weights = None
    self._learn_rate = tf.compat.v1.train.piecewise_constant(
        self.step, list(np.array(opt_config.lr_decay_steps).astype(np.int64)),
        list(opt_config.lr_list))
    opt = self._get_optimizer(opt_config.optimizer, self.learn_rate)
    self._optimizer = opt

  def run_backbone(self, x, is_training=tf.constant(True)):
    """Run backbone.

    Args:
      x: [B, T, ...] B: mini-batch, T: episode length.
      is_training: Bool. Whether in training mode.
    Returns:
      h: [B, T, D] D: feature length.
    """
    x_shape = x.shape
    if len(x_shape) == 5:
      assert x_shape[0] == 1
      x = tf.squeeze(x, 0)
    return self.backbone(x, is_training=is_training)

  def forward(self, x, y, x_test, is_training=tf.constant(True)):
    """Run forward pass."""
    h = self.run_backbone(x, is_training=is_training)
    num_classes = tf.reduce_max(y) + 1
    if len(y.shape) == 2:
      assert y.shape[0] == 1
      y = tf.squeeze(y, 0)
    y_ = tf.expand_dims(tf.one_hot(y, num_classes), 1)  # [N, 1, K]
    h_ = tf.expand_dims(h, 2)  # [N, D, 1]
    nshot = tf.reduce_sum(y_, [0, 1])
    protos = tf.reduce_sum(y_ * h_, [0], keepdims=True) / nshot  # [1, D, K]
    h_test = self.run_backbone(x_test, is_training=is_training)  # [M, D]
    h_test_ = tf.expand_dims(h_test, 2)
    logits = -tf.reduce_sum((h_test_ - protos)**2, [1])  # [M, K]
    return logits

  @tf.function
  def train_step(self, x, y, x_test, y_test):
    """One training step."""
    # Calculates gradients
    with tf.GradientTape() as tape:
      logits = self.forward(x, y, x_test, is_training=True)
      xent = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=tf.reshape(y_test, [-1])))
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      loss = xent + reg_loss * self.wd
      var_list = self.var_to_optimize()
      grad_list = tape.gradient(loss, var_list)
      self._step.assign_add(1)
    opt = self.optimizer
    opt.apply_gradients(zip(grad_list, var_list))
    return xent

  @tf.function
  def eval_step(self, x, y, x_test):
    """One evaluation step."""
    prediction = self.forward(x, y, x_test, is_training=False)
    return prediction

  def get_var_to_optimize(self):
    """gets the list of variables to optimize."""
    var = super(ProtoNet, self).get_var_to_optimize()
    if self.config.protonet_config.freeze_backbone:
      bb_var = set(self.backbone.weights)
      var = list(filter(lambda x: x not in bb_var, var))
    return var

  @property
  def learn_rate(self):
    return self._learn_rate

  @property
  def step(self):
    return self._step

  @property
  def backbone(self):
    return self._backbone

  @property
  def wd(self):
    return self._wd

  @property
  def config(self):
    return self._config
