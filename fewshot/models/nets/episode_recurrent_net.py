"""A recurrent net base class for episodic learning.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nets.net import Net
from fewshot.utils.logger import get as get_logger

log = get_logger()


class EpisodeRecurrentNet(Net):
  """A recurrent net base class for episodic learning. This network takes in
  one example at a time.
  """

  def __init__(self, config, backbone, distributed=False, dtype=tf.float32):
    super(EpisodeRecurrentNet, self).__init__(dtype=dtype)
    self._config = config
    self._backbone = backbone
    self._wd = backbone.config.weight_decay
    self._distributed = distributed
    opt_config = self.config.optimizer_config
    gs = self._get_variable(
        'step', lambda: tf.zeros([], dtype=tf.int64), trainable=False)
    self._step = gs
    self._regularized_weights = None
    self._regularizer_wd = None
    self._decay_list = list(
        np.array(opt_config.lr_decay_steps).astype(np.int64))
    self._max_train_steps = opt_config.max_train_steps
    self._learn_rate = tf.compat.v1.train.piecewise_constant(
        self.step, self._decay_list, list(opt_config.lr_list))
    self._optimizer = self._get_optimizer(opt_config.optimizer,
                                          self.learn_rate)
    self._ssl_store = None
    if self.config.set_backbone_lr:
      self._var_to_optimize_2 = None
      log.info('Set a different optimizer for backbone')
      log.info('Backbone LR multiplier = {:.3f}'.format(
          self.config.backbone_lr_multiplier))
      bb_learn_rate = tf.compat.v1.train.piecewise_constant(
          self.step, list(
              np.array(opt_config.lr_decay_steps).astype(np.int64)),
          list(
              map(lambda x: x * self.config.backbone_lr_multiplier,
                  opt_config.lr_list)))
      self._bb_optimizer = self._get_optimizer(opt_config.optimizer,
                                               bb_learn_rate)

  def slice_time(self, x, t):
    """Gets the t-th timestep input."""
    return x[:, t]

  def run_backbone(self, x, is_training=tf.constant(True)):
    """Run backbone.

    Args:
      x: [B, T, ...] B: mini-batch, T: episode length.
      is_training: Bool. Whether in training mode.
    Returns:
      h: [B, T, D] D: feature length.
    """
    x_shape = tf.shape(x)
    new_shape = tf.concat([[x_shape[0] * x_shape[1]], x_shape[2:]], axis=0)
    x = tf.reshape(x, new_shape)
    x = self.backbone(x, is_training=is_training)
    h_shape = tf.shape(x)
    old_shape = tf.concat([x_shape[:2], h_shape[1:]], axis=0)
    x = tf.reshape(x, old_shape)
    return x

  def compute_loss(self, logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

  @tf.function
  def train_step(self,
                 x,
                 y,
                 s=None,
                 y_gt=None,
                 flag=None,
                 x_test=None,
                 y_test=None,
                 flag_test=None,
                 **kwargs):
    """One training step.

    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.
      y_gt: [B, T], groundtruth at each timestep, if different from labels.
      x_test: [B, M, ...], inputs of the query set, optional.
      y_test: [B, M], groundtruth of the query set, optional.

    Returns:
      xent: Cross entropy loss.
    """
    # tf.print('y', y[0], summarize=100)
    if self._distributed:
      import horovod.tensorflow as hvd
    if y_gt is None:
      y_gt = y
    with tf.GradientTape() as tape:
      if x_test is not None:
        # Additional query set (optional).
        assert y_test is not None
        logits, logits_test = self.forward(
            x, y, s=s, x_test=x_test, is_training=tf.constant(True))
        logits_all = tf.concat([logits, logits_test], axis=1)  # [B, T+N, Kmax]
        labels_all = tf.concat([y_gt, y_test], axis=1)  # [B, T+N]
      else:
        logits = self.forward(x, y, s=s, is_training=tf.constant(True))
        logits_all = logits
        labels_all = y_gt

      xent = self.compute_loss(logits_all, labels_all)

      # Cross entropy loss.
      if flag is not None:
        if flag_test is not None:
          flag_all = tf.concat([flag, flag_test], axis=1)
        else:
          flag_all = flag
        flag_ = tf.cast(flag_all, self.dtype)
        valid_sum = tf.reduce_sum(flag_)
        delta = tf.cast(tf.equal(valid_sum, 0.0), self.dtype)
        xent = tf.reduce_sum(xent * flag_) / (valid_sum + delta)
      else:
        xent = tf.reduce_mean(xent)

      # Regularizers.
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      loss = xent + reg_loss * self.wd

    # Apply gradients.
    if self._distributed:
      tape = hvd.DistributedGradientTape(tape)

    self.apply_gradients(loss, tape)

    return xent

  def apply_gradients(self, loss, tape, add_step=tf.constant(True)):
    """Apply gradients to optimizers."""
    if not self.config.set_backbone_lr:
      # Regular training, same learning rate for all.
      var_list = self.var_to_optimize()
      grad_list = tape.gradient(loss, var_list)
      if add_step:
        self._step.assign_add(1)
        # for g, v in zip(grad_list, var_list):
        #   if g is not None:
        #     tf.print([v.name, tf.reduce_mean(g)])
        # gall = tf.add_n([
        #     tf.reduce_mean(g) for g in filter(lambda g: g is not None, grad_list)
        # ])
        # if not tf.math.is_nan(gall):
      self.optimizer.apply_gradients(zip(grad_list, var_list))
      # else:
      #   tf.print('NaN in gradient detected!')
    else:
      # Separate the variables into two sets.
      if add_step:
        self._step.assign_add(1)
      bb_var_list, not_bb_var_list = self.var_to_optimize_2()
      grad_list = tape.gradient(loss, bb_var_list + not_bb_var_list)
      bb_grad_list = grad_list[:len(bb_var_list)]
      not_bb_grad_list = grad_list[len(bb_var_list):]
      bb = zip(bb_grad_list, bb_var_list)
      not_bb = zip(not_bb_grad_list, not_bb_var_list)
      if add_step:
        self._step.assign_add(1)
      self.optimizer.apply_gradients(not_bb)
      self._bb_optimizer.apply_gradients(bb)

  def get_var_to_optimize_2(self):
    """Gets the list of variables to optimize, separate backbone vs.
    non-backbone."""
    var_list = self.var_to_optimize()
    var_names = list(map(lambda x: x.name, var_list))
    var_dict = dict(zip(var_names, var_list))
    bb_var_list = list(filter(lambda x: x.trainable, self.backbone.weights()))
    bb_var_names = list(map(lambda x: x.name, bb_var_list))
    bb_var_names_set = set(bb_var_names)
    not_bb_var_names = list(
        filter(lambda vname: vname not in bb_var_names_set, var_names))
    not_bb_var_list = list(
        map(lambda vname: var_dict[vname], not_bb_var_names))
    return bb_var_list, not_bb_var_list

  def var_to_optimize_2(self):
    """Gets the list of variables to optimize, separate backbone vs.
    non-backbone."""
    if self._var_to_optimize_2 is None:
      self._var_to_optimize_2 = self.get_var_to_optimize_2()
      for v in self._var_to_optimize_2[0]:
        log.info("Trainable backbone weight {}".format(v.name))
      for v in self._var_to_optimize_2[1]:
        log.info("Trainable non-backbone weight {}".format(v.name))
      log.info('Num trainable weight: {}'.format(
          len(self._var_to_optimize_2[0]) + len(self._var_to_optimize_2[1])))
    return self._var_to_optimize_2

  @tf.function
  def eval_step(self, x, y, s=None, x_test=None, **kwargs):
    """One evaluation step.
    Args:
      x: [T, ...], inputs at each timestep.
      y: [T], label at each timestep.
      x_test: [M, ...], inputs of the query set, optional.

    Returns:
      logits: [T, Kmax], prediction.
      logits_test: [M, Kmax], prediction on the query set, optional.
    """
    if x_test is not None:
      logits, logits_test = self.forward(
          x, y, s=s, x_test=x_test, is_training=tf.constant(False))
      return logits, logits_test
    else:
      logits = self.forward(x, y, s=s, is_training=tf.constant(False))
      return logits

  def predict_id(self, logits):
    return tf.argmax(logits, axis=-1)

  @property
  def config(self):
    """Config"""
    return self._config

  @property
  def backbone(self):
    """Backbone"""
    return self._backbone

  @property
  def wd(self):
    """Weight decay coefficient"""
    return self._wd

  @property
  def learn_rate(self):
    """Learning rate"""
    return self._learn_rate

  @property
  def step(self):
    """Step"""
    return self._step

  @property
  def max_train_steps(self):
    return self._max_train_steps
