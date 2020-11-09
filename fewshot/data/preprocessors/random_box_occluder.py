"""Occluding the image with a random box.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.data.preprocessors.preprocessor import Preprocessor


class RandomBoxOccluder(Preprocessor):

  @tf.function
  def preprocess(self, inputs):
    """NHWC float format."""
    image = tf.image.convert_image_dtype(inputs, tf.float32)
    N = inputs.shape[0]
    W = inputs.shape[2]
    H = inputs.shape[1]
    C = inputs.shape[3]
    BW = int(W * 0.3)
    BH = int(H * 0.3)
    box_loc = tf.cast(
        tf.floor(
            tf.random.uniform([N, 2]) *
            tf.constant([H - BH, W - BW], dtype=tf.float32)), tf.int32)
    w_range = tf.reshape(tf.range(BW), [1, 1, -1, 1])
    h_range = tf.reshape(tf.range(BH), [1, -1, 1, 1])
    box_idx = tf.concat(
        [tf.tile(h_range, [1, 1, BW, 1]),
         tf.tile(w_range, [1, BH, 1, 1])],
        axis=-1)
    box_idx += tf.reshape(box_loc, [N, 1, 1, 2])
    Nidx = tf.tile(tf.reshape(tf.range(N), [-1, 1, 1, 1]), [1, BW, BH, 1])
    box_idx = tf.concat([Nidx, box_idx], axis=-1)
    box_idx = tf.reshape(box_idx, [N * BW * BH, 3])
    mask = tf.scatter_nd(box_idx, tf.ones([N * BW * BH, C]), tf.shape(inputs))
    image = image * (1.0 - mask) + 0.5 * mask
    return image
