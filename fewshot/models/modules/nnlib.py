"""Basic modules.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.weight_module import WeightModule
from fewshot.models.variable_context import variable_scope


class Conv2D(WeightModule):
  """2D convolution layer."""

  def __init__(self,
               name,
               filter_size,
               in_filters,
               out_filters,
               strides,
               data_format='NCHW',
               dtype=tf.float32,
               wdict=None):
    super(Conv2D, self).__init__(dtype=dtype)
    with variable_scope(name):
      kinit = self._get_normal_init(
          [filter_size, filter_size, in_filters, out_filters])
      self._kernel = self._get_variable("w", kinit, dtype=dtype, wdict=wdict)
    self._strides = strides
    self._data_format = data_format

  def forward(self, x, **kwargs):
    return tf.nn.conv2d(
        x,
        self._kernel,
        self._strides,
        padding="SAME",
        data_format=self._data_format)


class Linear(WeightModule):
  """Fully connected layer."""

  def __init__(self,
               name,
               in_dim,
               out_dim,
               w_init=None,
               b_init=None,
               add_bias=True,
               dtype=tf.float32,
               wdict=None):
    super(Linear, self).__init__(dtype=dtype)
    if w_init is None:
      w_init = self._get_uniform_init(in_dim, out_dim)
    if b_init is None:
      b_init = self._get_constant_init([out_dim], 0.0)
    self._add_bias = add_bias
    with variable_scope(name):
      self._weight = self._get_variable("w", w_init, wdict=wdict)
      if add_bias:
        self._bias = self._get_variable("b", b_init, wdict=wdict)
    self._in_dim = in_dim
    self._out_dim = out_dim
    self._name = name

  def forward(self, x, **kwargs):
    z = tf.matmul(x, self._weight)
    if self._add_bias:
      z += self._bias
    return z


class CosineLinear(WeightModule):
  """Fully connected layer."""

  def __init__(self,
               name,
               in_dim,
               out_dim,
               w_init=None,
               temp=None,
               learn_temp=False,
               dtype=tf.float32,
               wdict=None):
    super(CosineLinear, self).__init__(dtype=dtype)
    if w_init is None:
      w_init = self._get_uniform_init(in_dim, out_dim)
    with variable_scope(name):
      self._weight = self._get_variable("w", w_init, wdict=wdict)
    self._in_dim = in_dim
    self._out_dim = out_dim
    self._name = name
    if not learn_temp:
      self._temp = temp
    else:
      self._temp = self._get_variable("temp", lambda: tf.zero([]) + temp)

  def forward(self, x, **kwargs):
    z = tf.matmul(x, self._weight)
    x_norm = tf.maximum(tf.norm(x, axis=-1, keepdims=True), 1e-7)
    w_norm = tf.maximum(tf.norm(self._weight, axis=0, keepdims=True), 1e-7)
    z = z / x_norm / w_norm
    if self._temp is not None:
      z *= self._temp
    return z


class BatchNorm(WeightModule):
  """Batch normalization layer."""

  def __init__(self,
               name,
               num_channels,
               data_format="NCHW",
               eps=1e-3,
               decay=0.999,
               dtype=tf.float32,
               wdict=None):
    super(BatchNorm, self).__init__(dtype=dtype)
    assert data_format in ["NCHW", "NHWC", "NHC", "NC"]
    if data_format == "NCHW":
      self._axis = 1
      self._axes = [0, 2, 3]
    elif data_format == "NHWC":
      self._axis = -1
      self._axes = [0, 1, 2]
    elif data_format == "NC":
      self._axis = -1
      self._axes = [0]
    elif data_format == "NHC":
      self._axis = -1
      self._axes = [0, 1]
    self._eps = eps
    self._decay = decay
    self._data_format = data_format
    C = num_channels
    with variable_scope(name):
      self._beta = self._get_variable(
          "beta", self._get_constant_init([C], 0.0), wdict=wdict)
      self._gamma = self._get_variable(
          "gamma", self._get_constant_init([C], 1.0), wdict=wdict)
      self._emean = self._get_variable(
          "moving_mean",
          self._get_constant_init([C], 0.0),
          trainable=False,
          wdict=wdict)
      self._evar = self._get_variable(
          "moving_variance",
          self._get_constant_init([C], 0.0),
          trainable=False,
          wdict=wdict)

  def _expand(self, v):
    return tf.reshape(v, [1, -1, 1, 1])

  def forward(self, x, is_training):
    if is_training:
      return self.train_forward(x)
    else:
      return self.eval_forward(x)

  def train_forward(self, x):
    if self._data_format == "NCHW":
      _gamma_ = self._expand(self._gamma)
      _beta_ = self._expand(self._beta)
    else:
      _gamma_ = self._gamma
      _beta_ = self._beta
    mean, var = tf.nn.moments(x, axes=self._axes, keepdims=True)

    self._emean.assign_sub(
        (self._emean - tf.squeeze(mean)) * (1 - self._decay))
    self._evar.assign_sub((self._evar - tf.squeeze(var)) * (1 - self._decay))
    return tf.nn.batch_normalization(x, mean, var, _beta_, _gamma_, self._eps)

  def eval_forward(self, x):
    if self._data_format == "NCHW":
      _gamma_ = self._expand(self._gamma)
      _beta_ = self._expand(self._beta)
      _emean_ = self._expand(self._emean)
      _evar_ = self._expand(self._evar)
    else:
      _gamma_ = self._gamma
      _beta_ = self._beta
      _emean_ = self._emean
      _evar_ = self._evar
    return tf.nn.batch_normalization(x, _emean_, _evar_, _beta_, _gamma_,
                                     self._eps)
