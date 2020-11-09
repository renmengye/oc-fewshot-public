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
    self._stride = stride
    with variable_scope(name):
      self._conv1 = Conv2D(
          "conv1",
          3,
          in_filter,
          out_filter,
          stride,
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
    self._data_format = data_format
    self._add_relu = True

    self._ds = stride[2] > 1
    self._pad = in_filter < out_filter
    self._pad_size = [(out_filter - in_filter) // 2,
                      (out_filter - in_filter) // 2]
    if data_format == "NCHW":
      self._pad_arr = [[0, 0], self._pad_size, [0, 0], [0, 0]]
    else:
      self._pad_arr = [[0, 0], [0, 0], [0, 0], self._pad_size]
    # self._ds = stride[2] > 1 or in_filter != out_filter
    # if self._ds:
    #   self._projconv = Conv2D(
    #       "projconv",
    #       1,
    #       in_filter,
    #       out_filter,
    #       stride,
    #       dtype=dtype)
    #   self._projbn = BatchNorm(
    #       "projbn", out_filter, dtype=dtype)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def _possible_downsample(self, x):
    """Downsample the feature map using average pooling, if the filter size
    does not match."""
    # if self._ds:
    #   return self._projbn(self._projconv(x))
    # else:
    #   return x
    if self._ds:
      x = tf.nn.avg_pool(
          x,
          self._stride,
          self._stride,
          padding="SAME",
          data_format=self._data_format)
    if self._pad:
      x = tf.pad(x, self._pad_arr)
    return x

  def forward(self, x, is_training, **kwargs):
    origx = x
    x = self._conv1(x)
    x = self._bn1(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = self._conv2(x)
    x = self._bn2(x, is_training=is_training)
    x += self._possible_downsample(origx)
    if self._add_relu:
      x = tf.nn.relu(x)
    return x


class BottleneckResidualModule(ContainerModule):

  def __init__(self,
               name,
               in_filter,
               out_filter,
               stride,
               data_format="NCHW",
               dtype=tf.float32,
               add_relu=True):
    super(BottleneckResidualModule, self).__init__()
    self._data_format = data_format
    self._stride = stride
    with variable_scope(name):
      self._conv1 = Conv2D(
          "conv1", 1, in_filter, out_filter // 4, stride, dtype=dtype)
      self._bn1 = BatchNorm("bn1", out_filter // 4, dtype=dtype)
      self._conv2 = Conv2D(
          "conv2",
          3,
          out_filter // 4,
          out_filter // 4,
          self._stride_arr(1),
          dtype=dtype)
      self._bn2 = BatchNorm("bn2", out_filter // 4, dtype=dtype)
      self._conv3 = Conv2D(
          "conv3",
          1,
          out_filter // 4,
          out_filter,
          self._stride_arr(1),
          dtype=dtype)
      self._bn3 = BatchNorm("bn3", out_filter, dtype=dtype)
      self._ds = stride[2] > 1 or in_filter != out_filter
      if self._ds:
        self._projconv = Conv2D(
            "projconv", 1, in_filter, out_filter, stride, dtype=dtype)
        self._projbn = BatchNorm("projbn", out_filter, dtype=dtype)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def forward(self, x, is_training, **kwargs):
    origx = x
    x = self._conv1(x)
    x = self._bn1(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = self._conv2(x)
    x = self._bn2(x, is_training=is_training)
    x = tf.nn.relu(x)
    x = self._conv3(x)
    x = self._bn3(x, is_training=is_training)
    if self._ds:
      x += self._projbn(self._projconv(origx), is_training=is_training)
    if self._add_relu:
      x = tf.nn.relu(x)
    return x


class InitConvModule(ContainerModule):

  def __init__(self,
               name,
               filter_size,
               in_filter,
               out_filter,
               max_pool=True,
               data_format="NCHW",
               dtype=tf.float32):
    super(InitConvModule, self).__init__()
    self._data_format = data_format
    self._max_pool = max_pool
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
    self._data_format = data_format

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def forward(self, x, is_training, **kwargs):
    x = self._conv(x)
    x = self._bn(x, is_training=is_training)
    x = tf.nn.relu(x)
    if self._max_pool:
      x = tf.nn.max_pool(
          x,
          self._stride_arr(3),
          self._stride_arr(2),
          padding="SAME",
          data_format=self._data_format)
    return x


@RegisterModule('resnet_backbone')
class ResnetBackbone(Backbone):

  def __init__(self, config, dtype=tf.float32):
    super(ResnetBackbone, self).__init__(config, dtype=dtype)
    strides = config.strides
    filters = [ff for ff in config.num_filters]  # Copy filter config.
    self._blocks = []
    self._init_conv = InitConvModule(
        "init",
        config.init_filter,
        config.num_channels,
        filters[0],
        max_pool=config.init_max_pool,
        data_format=config.data_format,
        dtype=dtype)
    self._blocks.append(self._init_conv)
    if config.use_bottleneck:
      filters = [ff * 4 for ff in filters]
    self._config = config
    nlayers = sum(config.num_residual_units)
    ss = 0
    ii = 0
    for ll in range(nlayers):
      # Residual unit configuration.
      if ii == 0:
        in_filter = filters[ss]
        stride = self._stride_arr(strides[ss])
      else:
        in_filter = filters[ss + 1]
        stride = self._stride_arr(1)
      out_filter = filters[ss + 1]

      # Build residual unit.
      prefix = "unit_{}_{}".format(ss + 1, ii)
      add_relu = True
      if not config.add_last_relu:
        if ll == nlayers - 1:
          add_relu = False

      if config.use_bottleneck:
        m = BottleneckResidualModule(
            prefix,
            in_filter,
            out_filter,
            stride,
            data_format=config.data_format,
            add_relu=add_relu,
            dtype=dtype)
      else:
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
    for m in self._blocks:
      x = m(x, is_training=is_training)
    if self.config.global_avg_pool:
      x = self._global_avg_pool(x)
    return x
