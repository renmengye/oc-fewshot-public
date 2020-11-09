"""A recurrent net base class using sigmoid for getting a unknown output.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_net import EpisodeRecurrentNet


class dummy_context_mgr():

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class EpisodeRecurrentSigmoidNet(EpisodeRecurrentNet):
  """Episode recurrent network with sigmoid output."""

  def wt_avg(self, x, wt):
    """Weighted average.

    Args:
      x: [N]. Input.
      wt: [N]. Weight per input.
    """
    wsum = tf.reduce_sum(wt)
    delta = tf.cast(tf.equal(wsum, 0), self.dtype)
    wsum = tf.cast(wsum, self.dtype)
    wt = tf.cast(wt, self.dtype)
    return tf.reduce_sum(x * wt) / (wsum + delta)

  def xent_with_unk(self, logits, labels, K, flag):
    """Cross entropy with unknown sigmoid output with some flag."""
    # Gets unknowns.
    labels_unk = tf.cast(tf.equal(labels, K), self.dtype)  # [B, T+T']
    flag_unk = flag
    flag *= 1.0 - labels_unk

    # Do not compute loss if we predict that this is unknown.
    if self.config.disable_loss_self_unk:
      pred = self.predict_id(logits)
      is_logits_unk = tf.cast(tf.equal(pred, K), self.dtype)  # [B, T+T']
      flag *= 1.0 - is_logits_unk

    logits_unk = logits[:, :, -1]  # [B, T+T']
    logits = logits[:, :, :-1]  # [B, T+T', Kmax]
    labels = tf.math.minimum(
        tf.cast(tf.shape(logits)[-1], tf.int64) - 1,
        tf.cast(labels, tf.int64))  # [Kmax]

    # Cross entropy loss, either softmax or sigmoid.
    assert self.config.loss_fn == "softmax"
    if self.config.loss_fn == "softmax":
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
    elif self.config.loss_fn == "sigmoid":
      labels_onehot = tf.one_hot(labels, K)
      xent = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits, labels=labels_onehot), [-1])

    # Binary cross entropy on unknowns.
    xent_unk = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_unk, labels=labels_unk)
    xent = self.wt_avg(xent, flag)
    xent_unk = self.wt_avg(xent_unk, flag_unk)
    return xent, xent_unk

  def xent(self, logits, labels, flag):
    """Cross entropy with some flag."""
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    xent = self.wt_avg(xent, flag)
    return xent

  def compute_loss(self,
                   x,
                   y,
                   y_gt,
                   s=None,
                   x_test=None,
                   y_test=None,
                   flag=None,
                   flag_test=None,
                   **kwargs):
    """Compute the training loss."""
    if x_test is not None:
      assert y_test is not None
      logits, logits_test = self.forward(
          x, y, s=s, x_test=x_test, is_training=tf.constant(True))
      logits_all = tf.concat([logits, logits_test], axis=1)  # [B, T+N, Kmax+1]
      labels_all = tf.concat([y_gt, y_test], axis=1)  # [B, T+N]
    else:
      logits = self.forward(x, y, s=s, is_training=tf.constant(True))
      logits_all = logits
      labels_all = y_gt
    K = self.config.num_classes
    if flag is None:
      flag_all = tf.ones(tf.shape(y), dtype=self.dtype)
    else:
      if flag_test is not None:
        flag_all = tf.concat([flag, flag_test], axis=1)
      else:
        flag_all = flag
      flag_all = tf.cast(flag_all, self.dtype)
    xent, xent_unk = self.xent_with_unk(
        logits_all, labels_all, K, flag=flag_all)

    # Regularizers.
    reg_loss = self._get_regularizer_loss(*self.regularized_weights())
    xent += self.config.unknown_loss * xent_unk
    loss = xent + reg_loss * self.wd
    return loss, {'xent': xent, 'xent_unk': xent_unk}

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
                 writer=None,
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
    with writer.as_default() if writer is not None else dummy_context_mgr(
    ) as gs:
      with tf.GradientTape() as tape:
        loss, metric = self.compute_loss(
            x,
            y,
            y_gt,
            s=s,
            flag=flag,
            x_test=x_test,
            y_test=y_test,
            flag_test=flag_test,
            **kwargs)

      # Data parallel training.
      if self._distributed:
        xent_sync = tf.reduce_mean(
            hvd.allgather(
                tf.zeros([1], dtype=tf.float32) + metric['xent'], name='xent'))
        tape = hvd.DistributedGradientTape(tape)
      else:
        xent_sync = metric['xent']

      # Apply gradients.
      # if not tf.math.is_nan(xent_sync):
      self.apply_gradients(loss, tape)

      write_flag = self._distributed and hvd.rank() == 0
      write_flag = write_flag or (not self._distributed)

      if write_flag and writer is not None:
        if tf.equal(
            tf.math.floormod(self._step + 1,
                             self.config.train_config.steps_per_log), 0):
          for name, val in metric.items():
            if name != 'xent':
              tf.summary.scalar(name, val, step=self._step + 1)

          if self._ssl_store is not None:
            tf.summary.scalar(
                'ssl write',
                tf.reduce_mean(tf.cast(self._ssl_store, tf.float32)),
                step=self._step + 1)
          writer.flush()
    return xent_sync

  def renormalize(self, logits):
    """Renormalize the logits, with the last dimension to be unknown. The rest
    of the softmax is then multiplied by 1 - sigmoid."""
    assert False
    logits_unk = logits[:, :, -1:]
    logits_rest = logits[:, :, :-1]
    logits_rest = tf.math.log_softmax(logits_rest)
    logits_unk_c = tf.math.log(1.0 - tf.math.sigmoid(logits_unk))
    logits_rest += logits_unk_c
    return tf.concat([logits_rest, tf.math.log_sigmoid(logits_unk)], axis=-1)

  @tf.function
  def eval_step(self, x, y, s=None, x_test=None, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.
      x_test: [B, M, ...], inputs of the query set, optional.

    Returns:
      logits: [B, T, Kmax], prediction.
      logits_test: [B, M, Kmax], prediction on the query set, optional.
    """
    # Note that the model itself is still a cascaded output model. It doesn't
    # mean that we renormalize the logits. But for evaluating accuracy purpose
    # renormalizing the logits makes sure that we stay the same as our original
    # evaluation methods.
    if x_test is not None:
      # Additional query set (optional).
      logits, logits_test = self.forward(
          x, y, s=s, x_test=x_test, is_training=tf.constant(False))
    else:
      logits = self.forward(x, y, s=s, is_training=tf.constant(False))
      return logits

  def predict_id(self, logits):
    """Predict class ID based on logits."""
    unk = tf.greater(logits[:, :, -1], 0.0)  # [B, T]
    non_unk = tf.cast(
        tf.argmax(logits[:, :, :-1], axis=-1), dtype=tf.int32)  # [B, T]
    final = tf.where(unk, self.config.num_classes, non_unk)
    return final
