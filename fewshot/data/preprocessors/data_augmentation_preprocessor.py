"""Data augmentation preprocessor.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from fewshot.data.preprocessors.preprocessor import Preprocessor
from fewshot.utils.logger import get as get_logger

log = get_logger()
RND_ROT_DEG = 5.0


class DataAugmentationPreprocessor(Preprocessor):
  """Preprocessor for data augmentation."""

  def __init__(self, image_size, crop_size, random_crop, random_flip,
               random_color, random_rotate):
    """Initialize data augmentation preprocessor.

    Args:
      image_size: Int. Size of the output image.
      crop_size: Int. Random crop size (assuming square).
      random_crop: Bool. Whether to apply random crop.
      random_flip: Bool. Whether to apply random flip.
      random_color: Bool. Whether to apply random color.
    """
    self._image_size = image_size
    self._crop_size = crop_size
    self._random_crop = random_crop
    self._random_flip = random_flip
    self._random_color = random_color
    self._random_rotate = random_rotate

  @tf.function
  def _data_augment(self, image):
    with tf.device('/cpu:0'):
      image = tf.image.convert_image_dtype(image, tf.float32)

      if self.random_crop:
        image = tf.compat.v1.image.resize_image_with_crop_or_pad(
            image, self.crop_size, self.crop_size)
        image = tf.compat.v1.image.random_crop(
            image, [self.image_size, self.image_size, image.shape[-1]])
      else:
        image = tf.compat.v1.image.resize_image_with_crop_or_pad(
            image, self.image_size, self.image_size)

      if self.random_rotate:
        angle = tf.random.uniform([], minval=-RND_ROT_DEG, maxval=RND_ROT_DEG)
        angle = angle / 180.0 * np.pi
        image = tfa.image.rotate(image, angle)

      if self.random_flip:
        # log.info("Apply random flipping")
        image = tf.compat.v1.image.random_flip_left_right(image)

      # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
      if self.random_color:
        image = tf.compat.v1.image.random_brightness(image,
                                                     max_delta=63. / 255.)
        image = tf.compat.v1.image.random_saturation(image,
                                                     lower=0.5,
                                                     upper=1.5)
        image = tf.compat.v1.image.random_contrast(image, lower=0.2, upper=1.8)

    return image

  @tf.function
  def data_augment_batch(self, images):
    with tf.device('/cpu:0'):
      images = tf.image.convert_image_dtype(images, tf.float32)
      if self.random_rotate:
        images2 = images
        angle = tf.random.uniform([tf.shape(images)[0]],
                                  minval=-RND_ROT_DEG,
                                  maxval=RND_ROT_DEG)
        # tf.print('angle1', angle)
        angle = angle / 180.0 * np.pi
        # tf.print('angle2', angle)
        images = tfa.image.rotate(images, angle)
        # tf.print('max', tf.reduce_max(images2), tf.reduce_min(images2),
        #          tf.reduce_max(images - images2))

      if self.random_crop:
        images = tf.image.resize_with_crop_or_pad(images, self.crop_size,
                                                  self.crop_size)
        images = tf.image.random_crop(images, [
            tf.shape(images)[0], self.image_size, self.image_size,
            tf.shape(images)[-1]
        ])
      else:
        images = tf.image.resize_with_crop_or_pad(images, self.image_size,
                                                  self.image_size)

      if self.random_flip:
        images = tf.image.random_flip_left_right(images)

      if self.random_color:
        images = tf.image.random_brightness(images, max_delta=63. / 255.)
        images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
        images = tf.image.random_contrast(images, lower=0.2, upper=1.8)
    return images

  # def _preprocess(self, image):
  #   return self._data_augment(image)

  def preprocess(self, inputs):
    return self.data_augment_batch(inputs)
    # output = []
    # if len(inputs.shape) == 4:
    #   for i in range(inputs.shape[0]):
    #     output.append(self._preprocess(inputs[i]))
    #   return tf.stack(output, axis=0)
    # elif len(inputs.shape) == 3:
    #   return self._preprocess(inputs[i])
    # else:
    #   raise ValueError('Unknown shape inputs.shape')

  @property
  def random_crop(self):
    return self._random_crop

  @property
  def random_flip(self):
    return self._random_flip

  @property
  def random_color(self):
    return self._random_color

  @property
  def random_rotate(self):
    return self._random_rotate

  @property
  def image_size(self):
    return self._image_size

  @property
  def crop_size(self):
    return self._crop_size
