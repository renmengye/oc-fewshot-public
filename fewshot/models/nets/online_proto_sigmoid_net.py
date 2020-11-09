"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("online_proto_sigmoid_net")
@RegisterModel("proto_mem_sigmoid_net")  # Legacy name
class OnlineProtoSigmoidNet(EpisodeRecurrentSigmoidNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineProtoSigmoidNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    self._memory = memory

  def forward(self,
              x,
              y,
              s=None,
              x_test=None,
              is_training=tf.constant(True),
              **kwargs):
    """Make a forward pass.
    Args:
      x: [B, T, ...]. Support examples at each timestep.
      y: [B, T]. Support labels at each timestep, note that the label is not
                 revealed until the next step.

    Returns:
      y_pred: [B, T, K+1], Logits at each timestep.
    """
    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    h = self.run_backbone(x, is_training=is_training)
    y_pred = tf.TensorArray(self.dtype, size=T)
    states = self.memory.get_initial_state(h.shape[0])

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
      x_ = self.slice_time(h, t)  # [B, ...]
      y_ = self.slice_time(y, t)  # [B]
      if s is None:
        s_ = None
      else:
        s_ = self.slice_time(s, t)
      y_pred_, states = self.memory.forward_one(
          x_,
          y_,
          t,
          *states,
          s=s_,
          add_new=tf.constant(True),
          is_training=is_training,
          ssl_store=ssl_store[:, t])
      y_pred = y_pred.write(t, y_pred_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    if x_test is not None:
      x_test = self.run_backbone(x_test, is_training=is_training)  # [B, N, D]
      y_test_pred = self.memory.retrieve_all(
          x_test, add_new=tf.constant(False))
      return y_pred, y_test_pred
    else:
      return y_pred

  @property
  def memory(self):
    """Memory module"""
    return self._memory


from fewshot.models.nets.episode_recurrent_sigmoid_trunc_net import EpisodeRecurrentSigmoidTruncNet  # NOQA


@RegisterModel("proto_mem_sigmoid_trunc_net")
class ProtoMemSigmoidTruncNet(EpisodeRecurrentSigmoidTruncNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(ProtoMemSigmoidTruncNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    self._memory = memory

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
      x: [B, T, ...]. Support examples at each timestep.
      y: [B, T]. Support labels at each timestep, note that the label is not
                 revealed until the next step.

    Returns:
      y_pred: [B, T, K+1], Logits at each timestep.
    """
    h = self.run_backbone(x, is_training=is_training)
    y_pred = tf.TensorArray(self.dtype, size=dt)
    if len(states) == 0:
      states = self.memory.get_initial_state(tf.shape(h)[0])
      cold_start = True
    else:
      cold_start = False
    for t in tf.range(dt):
      x_ = self.slice_time(h, t)  # [B, ...]
      y_ = self.slice_time(y, t)  # [B]
      y_pred_, states = self.memory.forward_one(
          x_,
          y_,
          t,
          *states,
          add_new=tf.constant(True),
          is_training=is_training)
      y_pred = y_pred.write(t, y_pred_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    if cold_start:
      return y_pred
    else:
      return y_pred, states

  @property
  def memory(self):
    """Memory module"""
    return self._memory
