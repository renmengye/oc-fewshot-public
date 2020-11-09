"""Transpose image data to NCHW.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.data.preprocessors.preprocessor import Preprocessor


class NCHWPreprocessor(Preprocessor):

  @tf.function
  def preprocess(self, inputs):
    tf.print(tf.shape(inputs))
    return tf.transpose(inputs, [0, 3, 1, 2])
