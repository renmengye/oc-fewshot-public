"""ResNet SNAIL backbone module.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.backbone import Backbone
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Conv2D, Linear, BatchNorm
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger
log = get_logger()


class ResidualSnailModule(ContainerModule):

  def __init__(self,
               name,
               in_filter,
               out_filter,
               stride,
               data_format="NCHW",
               leaky_relu=0.0,
               dtype=tf.float32):
    super(ResidualSnailModule, self).__init__()
    self._data_format = data_format
    self._stride = stride
    self._leaky_relu = leaky_relu
    with variable_scope(name):
      self._conv1 = Conv2D(
          "conv1", 3, in_filter, out_filter, self._stride_arr(1), dtype=dtype)
      self._bn1 = BatchNorm(
          "bn1", out_filter, dtype=dtype, data_format=data_format)
      self._conv2 = Conv2D(
          "conv2", 3, out_filter, out_filter, self._stride_arr(1), dtype=dtype)
      self._bn2 = BatchNorm(
          "bn2", out_filter, dtype=dtype, data_format=data_format)
      self._conv3 = Conv2D(
          "conv3", 1, out_filter, out_filter, self._stride_arr(1), dtype=dtype)
      self._bn3 = BatchNorm(
          "bn3", out_filter, dtype=dtype, data_format=data_format)
      self._ds = stride[2] > 1 or in_filter != out_filter
      if self._ds:
        self._projconv = Conv2D(
            "projconv",
            1,
            in_filter,
            out_filter,
            self._stride_arr(1),
            dtype=dtype)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _relu(self, x):
    if self._leaky_relu > 0.001:
      return tf.nn.leaky_relu(x, alpha=self._leaky_relu)
    else:
      return tf.nn.relu(x)

  def forward(self, x, is_training):
    origx = x
    x = self._conv1(x)
    x = self._bn1(x, is_training=is_training)
    x = self._relu(x)
    x = self._conv2(x)
    x = self._bn2(x, is_training=is_training)
    x = self._relu(x)
    x = self._conv3(x)
    x = self._bn3(x, is_training=is_training)
    x = self._relu(x)
    if self._ds:
      x = tf.nn.max_pool(
          x + self._projconv(origx),
          self._stride_arr(2),
          self._stride_arr(2),
          padding='SAME',
          data_format=self._data_format)
    # if is_training:
    #   x = tf.nn.dropout(x, 0.1)
    return x


class InitConvModule(ContainerModule):

  def __init__(self,
               name,
               filter_size,
               in_filter,
               out_filter,
               leaky_relu=0.0,
               max_pool=True,
               data_format="NCHW",
               dtype=tf.float32):
    super(InitConvModule, self).__init__()
    self._data_format = data_format
    self._max_pool = max_pool
    self._leaky_relu = leaky_relu
    with variable_scope(name):
      self._conv = Conv2D(
          "conv",
          filter_size,
          in_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype)
      self._bn = BatchNorm(
          "bn", out_filter, data_format=data_format, dtype=dtype)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _relu(self, x):
    if self._leaky_relu > 0.001:
      return tf.nn.leaky_relu(x, alpha=self._leaky_relu)
    else:
      return tf.nn.relu(x)

  def forward(self, x, is_training):
    x = self._conv(x)
    x = self._bn(x, is_training=is_training)
    x = self._relu(x)
    if self._max_pool:
      x = tf.nn.max_pool(
          x,
          self._stride_arr(3),
          self._stride_arr(2),
          padding="SAME",
          data_format=self._data_format)
    return x


class FinalConvModule(ContainerModule):

  def __init__(self,
               name,
               in_filters,
               num_hidden=384,
               out_filters=512,
               data_format="NCHW",
               dtype=tf.float32):
    super(FinalConvModule, self).__init__()
    self._data_format = data_format
    with variable_scope(name):
      self._bn1 = BatchNorm("bn1", in_filters, data_format=data_format)
      self._conv = Conv2D("conv", 1, in_filters, num_hidden,
                          self._stride_arr(1))
      self._bn2 = BatchNorm("bn2", num_hidden, data_format=data_format)
      self._fc = Linear("fc", num_hidden, out_filters)
    self._out_filters = out_filters

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _global_avg_pool(self, x, keepdims=False):
    if self._data_format == 'NCHW':
      return tf.reduce_mean(x, [2, 3], keepdims=keepdims)
    else:
      return tf.reduce_mean(x, [1, 2], keepdims=keepdims)

  def forward(self, x, is_training):
    x = self._global_avg_pool(x, keepdims=True)
    x = self._bn1(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = self._conv(x)
    if is_training:
      x = tf.nn.dropout(x, 0.1)
    x = self._bn2(x, is_training=is_training)
    x = tf.nn.relu(x)
    if self._data_format == 'NCHW':
      x = tf.squeeze(x, [2, 3])
    else:
      x = tf.squeeze(x, [1, 2])
    x = self._fc(x)
    if is_training:
      x = tf.nn.dropout(x, 0.1)
    return x

  @property
  def out_filters(self):
    return self._out_filters


@RegisterModule('resnet_snail_backbone')
class ResnetSnailBackbone(Backbone):

  def __init__(self, config, dtype=tf.float32):
    super(ResnetSnailBackbone, self).__init__(config, dtype=dtype)
    strides = config.strides
    filters = [ff for ff in config.num_filters]  # Copy filter config.
    self._blocks = []
    self._init_conv = InitConvModule(
        "init",
        config.init_filter,
        config.num_channels,
        filters[0],
        leaky_relu=config.leaky_relu,
        max_pool=config.init_max_pool,
        data_format=config.data_format,
        dtype=dtype)
    self._config = config
    self._blocks.append(self._init_conv)
    nlayers = sum(config.num_residual_units)
    ss = 0
    ii = 0
    for ll in range(nlayers):
      if ii == 0:
        in_filter = filters[ss]
        stride = self._stride_arr(strides[ss])
      else:
        in_filter = filters[ss + 1]
        stride = self._stride_arr(1)
      out_filter = filters[ss + 1]

      # Build residual unit.
      prefix = "unit_{}_{}".format(ss + 1, ii)
      self._blocks.append(
          ResidualSnailModule(
              prefix,
              in_filter,
              out_filter,
              stride,
              data_format=config.data_format,
              leaky_relu=config.leaky_relu,
              dtype=dtype))

      if (ii + 1) % config.num_residual_units[ss] == 0:
        ss += 1
        ii = 0
      else:
        ii += 1

    final_conv = FinalConvModule(
        "final", filters[-1], data_format=config.data_format)
    self._final_conv = final_conv
    self._blocks.append(final_conv)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self.config.data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def forward(self, x, is_training=tf.constant(True)):
    for m in self._blocks:
      x = m(x, is_training=is_training)
    return x
