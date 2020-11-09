"""Group normalization.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.weight_module import WeightModule
from fewshot.models.variable_context import variable_scope


class GroupNorm(WeightModule):
  """Group normalization layer."""

  def __init__(self,
               name,
               num_channels,
               num_groups,
               data_format="NCHW",
               eps=1e-6,
               dtype=tf.float32,
               wdict=None):
    super(GroupNorm, self).__init__(dtype=dtype)
    if data_format == "NCHW":
      channels_axis = 1
      reduction_axes = (2, 3)
    elif data_format == "NHWC":
      channels_axis = 3
      reduction_axes = (1, 2)
    moment_axes = [channels_axis + 1]
    for a in reduction_axes:
      if a > channels_axis:
        moment_axes.append(a + 1)
      else:
        moment_axes.append(a)
    self._moment_axes = moment_axes
    self._channels_axis = channels_axis
    self._reduction_axes = reduction_axes
    self._num_groups = num_groups
    self._num_channels = num_channels
    self._eps = eps
    self._data_format = data_format
    shape = [num_channels]
    with variable_scope(name):
      self._beta = self._get_variable(
          "beta", self._get_constant_init(shape, 0.0), wdict=wdict)
      self._gamma = self._get_variable(
          "gamma", self._get_constant_init(shape, 1.0), wdict=wdict)

  def forward(self, x, **kwargs):
    x_shape = list(tf.shape(x))
    axes_before_channel = x_shape[:self._channels_axis]
    axes_after_channel = x_shape[self._channels_axis + 1:]
    G = self._num_groups
    C = self._num_channels
    shape_after = axes_before_channel + [G, C // G] + axes_after_channel
    x_reshape = tf.reshape(x, shape_after)
    mean, variance = tf.nn.moments(x_reshape, self._moment_axes, keepdims=True)
    gain = tf.compat.v1.rsqrt(variance + self._eps)
    offset = -mean * gain
    beta_shape = [1, 1, 1, 1, 1]
    beta_shape[self._channels_axis] = G
    beta_shape[self._channels_axis + 1] = C // G
    beta_reshape = tf.reshape(self._beta, beta_shape)
    gamma_reshape = tf.reshape(self._gamma, beta_shape)
    gain *= gamma_reshape
    offset *= gamma_reshape
    offset += beta_reshape
    normed = x_reshape * gain + offset
    return tf.reshape(normed, x_shape)
