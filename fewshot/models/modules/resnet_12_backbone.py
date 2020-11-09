"""ResNet used in TADAM.

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


class ResidualModule(ContainerModule):

  def __init__(self,
               name,
               in_filter,
               out_filter,
               stride,
               data_format="NCHW",
               dtype=tf.float32,
               add_relu=True):
    super(ResidualModule, self).__init__()
    self._data_format = data_format
    self._name = name
    self._stride = stride
    with variable_scope(name):
      self._conv1 = Conv2D(
          "conv1",
          3,
          in_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype)
      self._bn1 = BatchNorm(
          "bn1", out_filter, data_format=data_format, dtype=dtype)
      self._conv2 = Conv2D(
          "conv2",
          3,
          out_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype)
      self._bn2 = BatchNorm(
          "bn2", out_filter, data_format=data_format, dtype=dtype)
      self._conv3 = Conv2D(
          "conv3",
          3,
          out_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype)
      self._bn3 = BatchNorm(
          "bn3", out_filter, data_format=data_format, dtype=dtype)
      self._projconv = Conv2D(
          self._prefix(name, "projconv"),
          1,
          in_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype)
      self._projbn = BatchNorm(
          self._prefix(name, "projbn"),
          out_filter,
          data_format=data_format,
          dtype=dtype)
    self._data_format = data_format
    self._add_relu = add_relu

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _possible_downsample(self, x):
    """Downsample the feature map using average pooling, if the filter size
    does not match."""
    if self._ds:
      x = tf.nn.max_pool(
          x,
          self._stride,
          self._stride,
          padding="SAME",
          data_format=self._data_format)
    if self._pad:
      x = tf.pad(x, self._pad_arr)
    return x

  def forward(self, x, is_training=True, **kwargs):
    origx = x
    x = self._conv1(x)
    x = self._bn1(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = self._conv2(x)
    x = self._bn2(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = self._conv3(x)
    x = self._bn3(x, is_training=is_training)
    shortcut = self._projbn(self._projconv(origx), is_training=is_training)

    x += shortcut

    x = tf.nn.max_pool(
        x,
        self._stride,
        self._stride,
        padding="SAME",
        data_format=self._data_format)

    if self._add_relu:
      x = tf.nn.relu(x)
    return x


@RegisterModule('resnet_12_backbone')
class Resnet12Backbone(Backbone):

  def __init__(self, config, dtype=tf.float32):
    super(Resnet12Backbone, self).__init__(config, dtype=dtype)
    strides = config.strides
    filters = [ff for ff in config.num_filters]  # Copy filter config.
    self._blocks = []
    self._config = config
    nlayers = sum(config.num_residual_units)
    ss = 0
    ii = 0
    for ll in range(nlayers):
      # Residual unit configuration.
      if ll == 0:
        in_filter = config.num_channels
        out_filter = filters[0]
        stride = self._stride_arr(strides[ss])
      else:
        if ii == 0:
          in_filter = filters[ss - 1]
          out_filter = filters[ss]
          stride = self._stride_arr(strides[ss])
        else:
          in_filter = filters[ss]
          out_filter = filters[ss]
          stride = self._stride_arr(1)

      # Build residual unit.
      prefix = "unit_{}_{}".format(ss + 1, ii)
      add_relu = True if ll < nlayers - 1 or config.add_last_relu else False
      log.info('{dd relu', add_relu)
      m = ResidualModule(
          prefix,
          in_filter,
          out_filter,
          stride,
          data_format=config.data_format,
          add_relu=add_relu,
          dtype=dtype)
      self._blocks.append(m)

      if (ii + 1) % config.num_residual_units[ss] == 0:
        ss += 1
        ii = 0
      else:
        ii += 1

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self.config.data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _global_avg_pool(self, x, keepdims=False):
    if self.config.data_format == 'NCHW':
      return tf.reduce_mean(x, [2, 3], keepdims=keepdims)
    else:
      return tf.reduce_mean(x, [1, 2], keepdims=keepdims)

  def forward(self, x, is_training=True, **kwargs):
    # tf.print(
    #     'input',
    #     [tf.shape(x),
    #      tf.reduce_mean(x),
    #      tf.reduce_max(x),
    #      tf.reduce_min(x)])
    for i, m in enumerate(self._blocks):
      x = m(x, is_training=is_training)

    # assert False, str(self.config.activation_scaling)
    if self.config.activation_scaling > 0.0:
      x *= self.config.activation_scaling
    if self.config.global_avg_pool:
      x = self._global_avg_pool(x)
    # if is_training:
    #   tf.print('x avgpool',
    #            [tf.reduce_mean(x),
    #             tf.reduce_max(x),
    #             tf.reduce_min(x)])
    if self.config.add_dropout and is_training:
      log.info('Apply droppout with rate {}'.format(self.config.dropout_rate))
      x = tf.nn.dropout(x, self.config.dropout_rate)
    return x
