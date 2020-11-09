"""A network for pretraining regular classification tasks.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.modules.nnlib import Linear
from fewshot.models.nets.net import Net
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("pretrain_net")
class PretrainNet(Net):

  def __init__(self, config, backbone, dtype=tf.float32):
    super(PretrainNet, self).__init__()
    self._backbone = backbone
    self._config = config
    assert self.config.num_classes > 0, 'Must specify number of output classes'
    opt_config = self.config.optimizer_config
    gs = tf.Variable(0, dtype=tf.int64, name='step', trainable=False)
    self._step = gs
    self._wd = backbone.config.weight_decay
    self._learn_rate = tf.compat.v1.train.piecewise_constant(
        self.step, list(np.array(opt_config.lr_decay_steps).astype(np.int64)),
        list(opt_config.lr_list))
    opt = self._get_optimizer(opt_config.optimizer, self.learn_rate)
    self._optimizer = opt
    out_dim = backbone.get_output_dimension()
    self._fc = Linear("fc", out_dim[-1], config.num_classes, dtype=dtype)

  def forward(self, x, is_training=tf.constant(True)):
    """Run forward pass."""
    h = self.backbone(x, is_training=is_training)
    logits = self._fc(h)
    return logits

  @tf.function
  def train_step(self, x, y):
    """One training step."""
    # Calculates gradients
    with tf.GradientTape() as tape:
      logits = self.forward(x, is_training=tf.constant(True))
      xent = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=y))
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      loss = xent + reg_loss * self.wd
      var_list = self.var_to_optimize()
      grad_list = tape.gradient(loss, var_list)
      self._step.assign_add(1)
    opt = self.optimizer
    opt.apply_gradients(zip(grad_list, var_list))
    return xent

  @tf.function
  def eval_step(self, x):
    """One evaluation step."""
    prediction = self.forward(x, is_training=tf.constant(False))
    return prediction

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
