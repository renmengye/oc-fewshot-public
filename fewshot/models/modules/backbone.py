"""Backbone base class.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule


class Backbone(ContainerModule):
  """Base classes for all backbones."""

  def __init__(self, config, dtype=tf.float32):
    super(Backbone, self).__init__()
    self._config = config
    self._dtype = dtype
    self._output_dim = None

  def get_dummy_input(self, batch_size=1):
    """Gets an all zero dummy input.

    Args:
      batch_size: batch size of the input.
    """
    H = self.config.height
    W = self.config.width
    C = self.config.num_channels
    dtype = self.dtype
    B = batch_size
    if self.config.data_format == 'NHWC':
      dummy_input = tf.zeros([B, H, W, C], dtype=dtype)
    else:
      dummy_input = tf.zeros([B, C, H, W], dtype=dtype)
    return dummy_input

  def get_output_dimension(self):
    """Get the output dimension of the backbone network.

    Returns:
      dim: List of integers, after the batch dimension.
    """
    if self._output_dim is None:
      dummy_input = self.get_dummy_input()
      dummy_output = self.forward(dummy_input, is_training=False)
      assert len(dummy_output.shape) > 1
      self._output_dim = [int(s) for s in dummy_output.shape[1:]]
    return self._output_dim

  @property
  def config(self):
    return self._config

  @property
  def dtype(self):
    return self._dtype
