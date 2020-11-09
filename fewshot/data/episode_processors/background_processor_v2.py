"""Processor that adds a random background.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf


class BackgroundProcessorV2(object):

  def __init__(self,
               background_dataset,
               random=True,
               random_apply=False,
               random_context=False,
               gaussian_noise_std=-1.0,
               apply_prob=1.0):
    """Build a background processor.

    Args:
      background_dataset:
      random: Whether to apply random cropping.
      random_apply: Whether to skip applying to certain ones.
      apply_prob: Probability of applying the background.
    """
    self._background_dataset = background_dataset
    self._random = random
    self._classmap = list(background_dataset.cls_dict.keys())
    self._rnd = np.random.RandomState(0)
    self._random_apply = random_apply
    self._random_context = random_context
    self._gaussian_noise_std = gaussian_noise_std
    self._apply_prob = apply_prob

  def process_one(self, x, bg):
    B = tf.shape(x)[0]
    crop_height = tf.shape(x)[1]
    crop_width = tf.shape(x)[2]
    num_channel = tf.shape(x)[-1]
    x = tf.image.convert_image_dtype(x, tf.float32)
    if self._random:
      bg = tf.image.random_crop(bg, [B, crop_height, crop_width, num_channel])
    else:
      bg = tf.image.resize_with_crop_or_pad(bg, crop_height, crop_width)

    # Additive gaussian noise.
    if self._gaussian_noise_std > 0.0:
      noise = tf.random.normal(tf.shape(bg), 0.0, self._gaussian_noise_std)
      bg = tf.minimum(tf.maximum(tf.add(bg, noise), 0.0), 1.0)

    bg = 1.0 - bg
    return tf.concat([x, bg], axis=-1)

  def __call__(self, episode):
    assert 'stage_id' in episode
    s = episode['stage_id']
    background_dataset = self.background_dataset
    classmap = self.classmap
    cls_dict = background_dataset.cls_dict
    # self.rnd.shuffle(classmap)
    if len(classmap) < s.max() + 1:
      replace = True
    else:
      replace = False
    if self._random_context:
      bgcls = self.rnd.choice(classmap, len(s), replace=True)
      image_ids = []
      for c in bgcls:
        rnd_idx = int(np.floor(self.rnd.uniform(0, len(cls_dict[c]))))
        image_ids.append(cls_dict[c][rnd_idx])
    else:
      classmap = self.rnd.choice(classmap, s.max() + 1, replace=replace)
      bgcls = classmap[s]
      image_ids = []
      for c in bgcls:
        rnd_idx = int(np.floor(self.rnd.uniform(0, len(cls_dict[c]))))
        image_ids.append(cls_dict[c][rnd_idx])

    image_ids = np.array(image_ids)
    bg = background_dataset.get_images(image_ids)
    bg = tf.image.convert_image_dtype(bg, tf.float32)
    bg = tf.reduce_mean(
        bg, [3], keepdims=True)  # [B, H, W, 1] # Assert grayscale.
    xs = episode['x_s']
    x_ = self.process_one(episode['x_s'], bg)
    episode['x_s'] = x_
    return episode

  @property
  def background_dataset(self):
    return self._background_dataset

  @property
  def classmap(self):
    return self._classmap

  @property
  def rnd(self):
    return self._rnd
