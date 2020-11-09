"""Basic 4-layer convolution network backbone.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.backbone import Backbone
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Conv2D, BatchNorm
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ConvModule(ContainerModule):

  def __init__(self,
               name,
               in_filter,
               out_filter,
               stride=2,
               add_relu=True,
               data_format="NCHW",
               pool_padding="SAME",
               dtype=tf.float32,
               wdict=None):
    super(ConvModule, self).__init__()
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
      self._bn = BatchNorm(
          "bn", out_filter, data_format=data_format, dtype=dtype, wdict=wdict)
    self._stride = stride
    self._add_relu = add_relu
    self._pool_padding = pool_padding

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def forward(self, x, is_training=tf.constant(True), **kwargs):
    # s = [int(s_) for s_ in x.shape]
    # for i in range(4):
    #   assert s[i] != 0, str(x.shape)
    x = self._conv(x)
    x = self._bn(x, is_training=is_training)
    # tf.print('hey', x.shape);
    if self._add_relu:
      x = tf.nn.relu(x)
    if self._stride > 1:
      x = tf.nn.max_pool(
          x,
          self._stride_arr(self._stride),
          self._stride_arr(self._stride),
          padding=self._pool_padding,
          data_format=self._data_format)
    return x


@RegisterModule("c4_backbone")
class C4Backbone(Backbone):

  def __init__(self, config, wdict=None):
    super(C4Backbone, self).__init__(config)
    self._config = config
    assert len(config.pool) == 0
    # assert config.add_last_relu
    if len(config.pool) > 0:
      pool = config.pool
    else:
      pool = [2, 2, 2, 2]
    self._conv1 = ConvModule(
        "conv1",
        config.num_channels,
        config.num_filters[0],
        stride=pool[0],
        pool_padding=config.pool_padding,
        data_format=config.data_format,
        wdict=wdict)
    self._conv2 = ConvModule(
        "conv2",
        config.num_filters[0],
        config.num_filters[1],
        stride=pool[1],
        pool_padding=config.pool_padding,
        data_format=config.data_format,
        wdict=wdict)
    self._conv3 = ConvModule(
        "conv3",
        config.num_filters[1],
        config.num_filters[2],
        stride=pool[2],
        pool_padding=config.pool_padding,
        data_format=config.data_format,
        wdict=wdict)
    self._conv4 = ConvModule(
        "conv4",
        config.num_filters[2],
        config.num_filters[3],
        stride=pool[3],
        add_relu=config.add_last_relu,
        pool_padding=config.pool_padding,
        data_format=config.data_format,
        wdict=wdict)

  def forward(self, x, is_training, **kwargs):
    for m in [self._conv1, self._conv2, self._conv3, self._conv4]:
      x = m(x, is_training=is_training)
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    if self.config.activation_scaling > 0:
      x = x * self.config.activation_scaling

    if self.config.add_dropout and is_training:
      log.info('Apply droppout with rate {}'.format(self.config.dropout_rate))
      x = tf.nn.dropout(x, self.config.dropout_rate)
    return x
