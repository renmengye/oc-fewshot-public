"""Online prototypical network.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_net import EpisodeRecurrentNet
from fewshot.models.registry import RegisterModel


@RegisterModel("online_proto_net")
@RegisterModel("proto_mem_net")  # Legacy name
class OnlineProtoNet(EpisodeRecurrentNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self, config, backbone, memory, dtype=tf.float32):
    super(OnlineProtoNet, self).__init__(config, backbone, dtype=dtype)
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
    T = tf.constant(x.shape[1])
    h = self.run_backbone(x, is_training=is_training)
    y_pred = tf.TensorArray(self.dtype, size=T)
    states = self.memory.get_initial_state(h.shape[0])
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
          is_training=is_training)
      y_pred = y_pred.write(t, y_pred_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    if x_test is not None:
      x_test = self.run_backbone(x_test, is_training=is_training)  # [B, N, D]
      y_test_pred = self.memory.retrieve_all(
          x_test, add_new=tf.constant(False))  # [B, N, Kmax]
      return y_pred, y_test_pred
    else:
      return y_pred

  @property
  def memory(self):
    """Memory module"""
    return self._memory
