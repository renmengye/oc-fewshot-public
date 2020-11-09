"""Layer normalization.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.weight_module import WeightModule
from fewshot.models.variable_context import variable_scope


class LayerNorm(WeightModule):
  """Layer normalization layer."""

  def __init__(self, name, num_channels, eps=1e-6, dtype=tf.float32):
    super(LayerNorm, self).__init__(dtype=dtype)
    shape = [num_channels]
    with variable_scope(name):
      binit = self._get_constant_init(shape, 0.0)
      ginit = self._get_constant_init(shape, 1.0)
      self._beta = self._get_variable("beta", binit)
      self._gamma = self._get_variable("gamma", ginit)
    self._eps = eps

  def forward(self, x):
    moment_axes = tf.range(1, len(x.shape))
    mean, variance = tf.nn.moments(x, moment_axes, keepdims=True)
    gain = tf.compat.v1.rsqrt(variance + self._eps)
    offset = -mean * gain
    gain *= self._gamma
    offset += self._beta
    normed = x * gain + offset
    return normed
