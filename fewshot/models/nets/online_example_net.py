"""Online example-based network.
The difference of this model with regular prototypical network is that it has
an example storage, rather than a prototype-based storage.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.registry import RegisterModel


@RegisterModel("online_example_net")
@RegisterModel("mixture_proto_net")  # Legacy name
class OnlineExampleNet(EpisodeRecurrentSigmoidNet):

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineExampleNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    self._memory = memory

  def forward(self, x, y, x_test=None, is_training=tf.constant(True),
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
      y_pred_, states = self.memory.forward_one(
          x_, y_, t, *states, is_training=is_training)
      y_pred = y_pred.write(t, y_pred_)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    if x_test is not None:
      assert False
    return y_pred

  @property
  def memory(self):
    """Memory module."""
    return self._memory
