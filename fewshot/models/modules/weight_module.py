"""A module base class that has weights in it.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.modules.module import Module
from fewshot.utils.logger import get as get_logger
from fewshot.models.variable_context import get_variable

log = get_logger()


class WeightModule(Module):

  def __init__(self, dtype=tf.float32):
    """Constructs a module object with some network configurations."""
    self._weights = []
    self._dtype = dtype
    self._regularizers = []

  def _get_variable(self, name, initializer, **kwargs):
    """Creates a learnable variable."""
    w = get_variable(name, initializer, **kwargs)
    self._weights.append(w)
    return w

  def _get_normal_init(self, shape):
    """Get a normal initializer for convolution kernels."""

    def _init():
      if len(shape) == 4:
        n = shape[0] * shape[1] * shape[3]
      elif len(shape) == 2:
        n = shape[1]
      elif len(shape) == 1:
        n = shape[0]
      winit = tf.random.truncated_normal(shape, 0.0,
                                         tf.sqrt(2.0 / tf.cast(n, tf.float32)))
      return winit

    return _init

  def _get_uniform_init(self, in_dim, out_dim):
    """Get a uniform initializer for fully connected layers."""

    def _init():
      factor = 1 / np.sqrt(float(out_dim))
      shape = [in_dim, out_dim]
      return tf.random.uniform(shape, -factor, factor)

    return _init

  def _get_constant_init(self, shape, val):
    """Get a constant initializer."""

    def _init():
      if val == 0.0:
        return tf.zeros(shape, dtype=self.dtype)
      elif val == 1.0:
        return tf.ones(shape, dtype=self.dtype)
      else:
        return tf.zeros(shape, dtype=self.dtype) + val

    return _init

  def _get_numpy_init(self, array):

    def _init():
      return tf.constant(array)

    return _init

  def weights(self):
    return self._weights

  def initialize(self, sess):
    sess.run(tf.compat.v1.initialize_variables(self.weights()))

  def set_trainable(self, trainable):
    for w in self.weights():
      w._trainable = trainable
      log.info('Set {} trainable={}'.format(w.name, trainable))

  @property
  def dtype(self):
    return self._dtype
