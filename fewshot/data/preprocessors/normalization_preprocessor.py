"""Normalization preprocessor.
Subtracts mean and divides by standard deviation.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
from fewshot.data.preprocessors.preprocessor import Preprocessor


class NormalizationPreprocessor(Preprocessor):
  """Normalization preprocessor, subtract mean and divide variance."""

  def __init__(self, mean=None, std=None):
    self._mean = mean
    self._std = std

  @tf.function
  def preprocess(self, inputs):
    if inputs.dtype == np.uint8 and type(inputs) == np.ndarray:
      inputs = inputs.astype(np.float32) / 255.0
    else:
      inputs = tf.image.convert_image_dtype(inputs, tf.float32)

    if self.mean is None or len(self.mean) == 0:
      inputs -= 0.5
    else:
      inputs -= self.mean

    if self.std is None or len(self.std) == 0:
      inputs /= 0.5
    else:
      inputs /= self.std
    return inputs

  @property
  def mean(self):
    return self._mean

  @property
  def std(self):
    return self._std
