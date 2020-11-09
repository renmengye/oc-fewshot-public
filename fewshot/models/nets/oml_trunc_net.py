"""Online meta-learning using truncated backpropagation through time.
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
from fewshot.utils.logger import get as get_logger

log = get_logger()


class dummy_context_mgr():

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


@RegisterModel("oml_trunc_sigmoid_net")
class OMLTruncSigmoidNet(EpisodeRecurrentSigmoidNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OMLTruncSigmoidNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory

    # Readjust training steps.
    opt_config = self.config.optimizer_config
    # ratio = config.num_steps
    # ratio = ratio // self.config.oml_config.inner_loop_truncate_steps
    self._max_train_steps = opt_config.max_train_steps  #// ratio

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

  def forward_partial(self, x, y, t_steps, *states, **kwargs):
    """Make a forward pass.

    Args:
      x: [B, T, ...]. Support examples.
      y: [B, T, ...]. Support examples labels.
      x_test: [B, T', ...]. Query examples.

    Returns:
      y_pred: [B, T]. Support example prediction.
      y_pred_test: [B, T']. Query example prediction, if exists.
    """
    B = tf.constant(x.shape[0])
    x = self.run_backbone(x, is_training=kwargs['is_training'])
    y_pred = tf.TensorArray(self.dtype, size=t_steps)
    # Maximum total number of classes.
    K = tf.constant(self.config.num_classes)
    # Current seen maximum classes.
    if len(states) == 0:
      states = self.memory.get_initial_state(B)
    for t in tf.range(t_steps):
      x_ = self.slice_time(x, t)  # [B, D]
      y_ = self.slice_time(y, t)  # [B]
      k = states[-1]
      y_cls_, y_unk_, states = self.memory(x_, y_, *states)
      # k = tf.maximum(k, tf.cast(y_, k.dtype))
      y_cls_ = self.mask(y_cls_, k, K)  # [B, NO]
      y_pred = y_pred.write(t, tf.concat([y_cls_, y_unk_[:, None]], axis=-1))
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    return y_pred, states

  def compute_loss(self, x, y, dt, y_gt, flag, *states, **kwargs):
    """Compute the training loss."""
    logits, states = self.forward_partial(
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

    # Regularizers.
    reg_loss = self._get_regularizer_loss(*self.regularized_weights())
    xent += self.config.unknown_loss * xent_unk
    loss = xent + reg_loss * self.wd

    # tf.print('y_gt', y_gt[0])
    # tf.print('logits', logits[0, :, :10], summarize=100)
    return loss, {'xent': xent, 'xent_unk': xent_unk}, states

  @tf.function
  def train_step(self, x, y, y_gt=None, flag=None, writer=None, **kwargs):
    """One training step, with truncated backpropagation through time.

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
    DT = self.config.oml_config.inner_loop_truncate_steps
    LOGSTEP = self.config.train_config.steps_per_log
    assert DT > 1
    with writer.as_default() if writer is not None else dummy_context_mgr(
    ) as gs:
      states = self.memory.get_initial_state(B)
      states_shape = [s.shape for s in states]
      xent_total = 0.0
      xent_unk_total = 0.0
      flag_total = tf.cast(tf.reduce_sum(flag), self.dtype)
      for t_start in tf.range(0, T, DT):
        # tf.print('t_start', t_start)
        with tf.GradientTape() as tape:
          # if tf.equal(t_start, 0):
          #   states = self.memory.get_initial_state(B)
          #   [s.set_shape(ss) for s, ss in zip(states, states_shape)]
          t_end = tf.minimum(t_start + DT, T)
          # tf.print('start', t_start, 'end', t_end)
          loss, metric, states = self.compute_loss(
              x[:, t_start:t_end], y[:, t_start:t_end], t_end - t_start,
              y_gt[:, t_start:t_end], flag[:, t_start:t_end], *states,
              **kwargs)

        if self._distributed:
          tape = hvd.DistributedGradientTape(tape)

        # Apply gradients.
        self.apply_gradients(loss, tape, add_step=tf.equal(t_start, 0))

        flag_total_ = tf.reduce_sum(
            tf.cast(flag[:, t_start:t_end], self.dtype))
        xent_total += metric['xent'] * flag_total_ / flag_total
        xent_unk_total += metric['xent_unk'] * flag_total_ / flag_total
      # tf.print('xent unk total', xent_unk_total)

      # Log xent unk
      if writer is not None:
        NSTEP = len(tf.range(0, T, DT))
        cond = tf.logical_or(
            tf.equal(tf.math.floormod(self._step, LOGSTEP), 0),
            tf.equal(self._step, 1))
        if cond:
          tf.summary.scalar('xent_unk', xent_unk_total, self._step)
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
    logits, _ = self.forward_partial(
        x, y, tf.shape(x)[1], is_training=tf.constant(False))
    return logits

  @property
  def memory(self):
    return self._memory

  def predict_id(self, logits):
    return tf.argmax(logits, axis=-1)


@RegisterModel("oml_trunc_net")
class OMLTruncNet(EpisodeRecurrentNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OMLTruncNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    assert config.fix_unknown, 'Only unknown is supported'
    self._memory = memory
    opt_config = self.config.optimizer_config
    # ratio = config.num_steps
    # ratio = ratio // self.config.oml_config.inner_loop_truncate_steps
    self._max_train_steps = opt_config.max_train_steps  #  // ratio

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

  def forward_partial(self, x, y, t_steps, *states, **kwargs):
    """Make a forward pass.

    Args:
      x: [B, T, ...]. Support examples.
      y: [B, T, ...]. Support examples labels.
      x_test: [B, T', ...]. Query examples.

    Returns:
      y_pred: [B, T]. Support example prediction.
      y_pred_test: [B, T']. Query example prediction, if exists.
    """
    B = tf.constant(x.shape[0])
    x = self.run_backbone(x, is_training=kwargs['is_training'])
    y_pred = tf.TensorArray(self.dtype, size=t_steps)
    # Maximum total number of classes.
    K = tf.constant(self.config.num_classes)
    # Current seen maximum classes.
    if len(states) == 0:
      states = self.memory.get_initial_state(B)
    for t in tf.range(t_steps):
      x_ = self.slice_time(x, t)  # [B, D]
      y_ = self.slice_time(y, t)  # [B]
      k = states[-1]
      y_cls_, y_unk_, states = self.memory(x_, y_, *states)
      y_cls_ = self.mask(y_cls_, k, K)  # [B, NO]
      y_pred = y_pred.write(t, tf.concat([y_cls_, y_unk_[:, None]], axis=-1))
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    return y_pred, states

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

  def compute_loss(self, x, y, dt, y_gt, flag, *states, **kwargs):
    """Compute the training loss."""
    logits, states = self.forward_partial(
        x, y, dt, *states, is_training=tf.constant(True))
    logits_all = logits
    labels_all = y_gt
    K = self.config.num_classes
    if flag is None:
      flag_all = tf.ones(tf.shape(y), dtype=self.dtype)
    else:
      flag_all = flag
      flag_all = tf.cast(flag_all, self.dtype)
    pred = tf.nn.softmax(logits_all)
    labels_onehot = tf.one_hot(labels_all, tf.shape(pred)[-1])  # [B, T, K]
    xent = -tf.math.log(tf.reduce_sum(pred * labels_onehot, [-1]))  # [B, T]
    xent = self.wt_avg(xent, flag_all)

    # Regularizers.
    reg_loss = self._get_regularizer_loss(*self.regularized_weights())
    loss = xent + reg_loss * self.wd
    return loss, {'xent': xent}, states

  @tf.function
  def train_step(self, x, y, y_gt=None, flag=None, writer=None, **kwargs):
    """One training step, with truncated backpropagation through time.

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
    DT = self.config.oml_config.inner_loop_truncate_steps
    LOGSTEP = self.config.train_config.steps_per_log
    assert DT > 1
    with writer.as_default() if writer is not None else dummy_context_mgr(
    ) as gs:
      states = self.memory.get_initial_state(B)
      xent_total = 0.0
      xent_unk_total = 0.0
      flag_total = tf.cast(tf.reduce_sum(flag), self.dtype)
      for t_start in tf.range(0, T, DT):
        with tf.GradientTape() as tape:
          t_end = tf.minimum(t_start + DT, T)
          loss, metric, states = self.compute_loss(
              x[:, t_start:t_end], y[:, t_start:t_end], t_end - t_start,
              y_gt[:, t_start:t_end], flag[:, t_start:t_end], *states,
              **kwargs)

        if self._distributed:
          tape = hvd.DistributedGradientTape(tape)

        # Apply gradients.
        self.apply_gradients(loss, tape, add_step=tf.equal(t_start, 0))
        flag_total_ = tf.reduce_sum(
            tf.cast(flag[:, t_start:t_end], self.dtype))
        xent_total += metric['xent'] * flag_total_ / flag_total
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
    logits, _ = self.forward_partial(
        x, y, tf.shape(x)[1], is_training=tf.constant(False))
    return logits

  @property
  def memory(self):
    return self._memory

  def predict_id(self, logits):
    return tf.argmax(logits, axis=-1)
