"""Basic 4-layer convolution network backbone.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.c4_backbone import C4Backbone
from fewshot.models.modules.backbone import Backbone
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope


@RegisterModule("c4_double_backbone")
class C4DoubleBackbone(Backbone):

  def __init__(self, config, wdict=None):
    super(C4DoubleBackbone, self).__init__(config)
    with variable_scope("b1"):
      self.b1 = C4Backbone(config)
    with variable_scope("b2"):
      self.b2 = C4Backbone(config)

  def forward(self, x, is_training, **kwargs):
    x1 = x[:, :, :, 0:1]
    x2 = x[:, :, :, 1:2]
    # tf.print('x1', tf.reduce_min(x1), tf.reduce_max(x1))
    # tf.print('x2', tf.reduce_min(x2), tf.reduce_max(x2))
    x1 = self.b1.forward(x1, is_training)
    x2 = self.b2.forward(x2, is_training)
    # assert False
    return tf.concat([x1, x2], axis=-1)

  def get_dummy_input(self, batch_size=1):
    """Gets an all zero dummy input.

    Args:
      batch_size: batch size of the input.
    """
    H = self.config.height
    W = self.config.width
    C = 2
    dtype = self.dtype
    B = batch_size
    if self.config.data_format == 'NHWC':
      dummy_input = tf.zeros([B, H, W, C], dtype=dtype)
    else:
      dummy_input = tf.zeros([B, C, H, W], dtype=dtype)
    return dummy_input
