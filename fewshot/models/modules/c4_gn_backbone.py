"""Basic 4-layer convolution network backbone, with group norm.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.backbone import Backbone
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Conv2D
from fewshot.models.modules.group_norm import GroupNorm
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope


class ConvGNModule(ContainerModule):

  def __init__(self,
               name,
               in_filter,
               out_filter,
               num_groups,
               stride=2,
               data_format="NCHW",
               dtype=tf.float32,
               wdict=None):
    super(ConvGNModule, self).__init__()
    self._data_format = data_format
    with variable_scope(name):
      self._conv = Conv2D(
          "conv",
          3,
          in_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype,
          wdict=wdict)
      self._gn = GroupNorm(
          "gn",
          out_filter,
          num_groups,
          data_format=data_format,
          dtype=dtype,
          wdict=wdict)
    self._stride = stride

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def forward(self, x, is_training=True, **kwargs):
    # print(x.shape)
    s = [int(s_) for s_ in x.shape]
    for i in range(4):
      assert s[i] != 0, str(x.shape)
    x = self._conv(x)
    x = self._gn(x, is_training=is_training)
    x = tf.nn.relu(x)
    if self._stride > 1:
      x = tf.nn.max_pool(
          x,
          self._stride_arr(self._stride),
          self._stride_arr(self._stride),
          padding="SAME",
          data_format=self._data_format)
    return x


@RegisterModule("c4_gn_backbone")
class C4GNBackbone(Backbone):

  def __init__(self, config, wdict=None):
    super(C4GNBackbone, self).__init__(config)
    self._config = config
    self._conv1 = ConvGNModule(
        "conv1",
        config.num_channels,
        config.num_filters[0],
        config.num_groups[0],
        data_format=config.data_format,
        wdict=wdict)
    self._conv2 = ConvGNModule(
        "conv2",
        config.num_filters[0],
        config.num_filters[1],
        config.num_groups[1],
        data_format=config.data_format,
        wdict=wdict)
    self._conv3 = ConvGNModule(
        "conv3",
        config.num_filters[1],
        config.num_filters[2],
        config.num_groups[2],
        data_format=config.data_format,
        wdict=wdict)
    self._conv4 = ConvGNModule(
        "conv4",
        config.num_filters[2],
        config.num_filters[3],
        config.num_groups[3],
        data_format=config.data_format,
        wdict=wdict)

  def forward(self, x, is_training, **kwargs):
    for m in [self._conv1, self._conv2, self._conv3, self._conv4]:
      x = m(x)
    x = tf.reshape(x, [x.shape[0], -1])
    return x
