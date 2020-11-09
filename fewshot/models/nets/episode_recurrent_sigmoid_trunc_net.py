"""A recurrent net base class using sigmoid for getting a unknown output.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA


class dummy_context_mgr():

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class EpisodeRecurrentSigmoidTruncNet(EpisodeRecurrentSigmoidNet):
  """Episode recurrent network with sigmoid output."""

  def __init__(self, config, backbone, distributed=False, dtype=tf.float32):
    super(EpisodeRecurrentSigmoidTruncNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    opt_config = self.config.optimizer_config
    ratio = config.num_steps
    ratio = ratio // self.config.optimizer_config.inner_loop_truncate_steps
    self._ratio = ratio
    self._max_train_steps = opt_config.max_train_steps // ratio

  def compute_loss(self, x, y, y_gt, flag, dt, *states, **kwargs):
    """Compute the training loss."""
    logits, states = self.forward(
        x, y, dt, *states, is_training=tf.constant(True))
    logits_all = logits
    labels_all = y_gt
    K = self.config.num_classes
    if flag is None:
      flag_all = tf.ones(tf.shape(y), dtype=self.dtype)
    else:
      flag_all = flag
      flag_all = tf.cast(flag_all, self.dtype)
    xent, xent_unk = self.xent_with_unk(
        logits_all, labels_all, K, flag=flag_all)
    # tf.print('hey2', 'xent', xent, 'xent unk', xent_unk)
    reg_loss = self._get_regularizer_loss(*self.regularized_weights())
    xent += self.config.unknown_loss * xent_unk
    loss = xent + reg_loss * self.wd
    return loss, {'xent': xent, 'xent_unk': xent_unk}, states

  @tf.function
  def train_step(self,
                 x,
                 y,
                 y_gt=None,
                 flag=None,
                 writer=None,
                 first_batch=False,
                 **kwargs):
    """One training step.

    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep, to be fed as input.
      y_unk: [B, T], binary label indicating unknown, used as groundtruth.
      y_gt: [B, T], groundtruth at each timestep, if different from labels.
      x_test: [B, M, ...], inputs of the query set, optional.
      y_test: [B, M], groundtruth of the query set, optional.

    Returns:
      xent: Cross entropy loss.
    """
    if self._distributed:
      import horovod.tensorflow as hvd
    if y_gt is None:
      y_gt = y
    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    with writer.as_default() if writer is not None else dummy_context_mgr(
    ) as gs:
      states = self.memory.get_initial_state(B, 64)
      DT = self.config.optimizer_config.inner_loop_truncate_steps
      # Data parallel training.
      xent_total = 0.0
      xent_unk_total = 0.0
      flag_total = tf.cast(tf.reduce_sum(flag), self.dtype)
      for t_start in range(0, self.config.num_steps, DT):
        t_end = tf.minimum(t_start + DT, T)
        with tf.GradientTape() as tape:
          loss, metric, states = self.compute_loss(
              x[:, t_start:t_end], y[:, t_start:t_end], y_gt[:, t_start:t_end],
              flag[:, t_start:t_end], t_start, DT, *states, **kwargs)

        # Apply gradients.
        if self._distributed:
          tape = hvd.DistributedGradientTape(tape)
        self.apply_gradients(loss, tape)

        # Sync weights initialization.
        if self._distributed and first_batch and tf.equal(t_start, 0):
          hvd.broadcast_variables(self.var_to_optimize(), root_rank=0)
          hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
          if self.config.set_backbone_lr:
            hvd.broadcast_variables(
                self._bb_optimizer.variables(), root_rank=0)

        flag_total_ = tf.reduce_sum(
            tf.cast(flag[:, t_start:t_end], self.dtype))
        xent_total += metric['xent'] * flag_total_ / flag_total
        xent_unk_total += metric['xent_unk'] * flag_total_ / flag_total

      write_flag = self._distributed and hvd.rank() == 0
      write_flag = write_flag or (not self._distributed)
      if write_flag and writer is not None:
        if tf.equal(
            tf.math.floormod(self._step // self._ratio + 1,
                             self.config.train_config.steps_per_log), 0):
          tf.summary.scalar('xent_unk', xent_unk_total, step=self._step + 1)
          writer.flush()
    return xent_total

  @tf.function
  def eval_step(self, x, y, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.

    Returns:
      logits: [B, T, Kmax], prediction.
    """
    logits = self.forward(
        x, y, 0, self.config.num_steps, is_training=tf.constant(False))
    return logits
