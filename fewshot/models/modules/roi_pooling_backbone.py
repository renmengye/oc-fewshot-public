"""ROI pooling layer.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.backbone import Backbone


@RegisterModule('roi_pooling_backbone')
class ROIPoolingBackbone(Backbone):

  def __init__(self, config, backbone, dtype=tf.float32):
    super(ROIPoolingBackbone, self).__init__(config, dtype=dtype)
    self._backbone = backbone

  def get_dummy_input(self, batch_size=1):
    """Gets an all zero dummy input.

    Args:
      batch_size: batch size of the input.
    """
    H = self.config.height
    W = self.config.width
    C = self.config.num_channels + 1
    dtype = self.dtype
    B = batch_size
    dummy_input = tf.zeros([B, H, W, C], dtype=dtype)
    return dummy_input

  def _global_avg_pool(self, x, keepdims=False):
    if self.config.data_format == 'NCHW':
      return tf.reduce_mean(x, [2, 3], keepdims=keepdims)
    else:
      return tf.reduce_mean(x, [1, 2], keepdims=keepdims)

  def _mask_pool(self, x, mask, keepdims=False):
    if self.config.data_format == 'NCHW':
      denom = tf.reduce_sum(mask, [2, 3])
      denom = tf.where(tf.equal(denom, 0), 1e-3, denom)
      return tf.reduce_sum(x * mask, [2, 3], keepdims=keepdims) / denom
    else:
      denom = tf.reduce_sum(mask, [1, 2])
      denom = tf.where(tf.equal(denom, 0), 1e-3, denom)
      return tf.reduce_sum(x * mask, [1, 2], keepdims=keepdims) / denom

  def forward(self, x, is_training=True, **kwargs):
    # tf.print(tf.shape(x))
    # assert self.config.data_format == "NCHW"
    mask = x[:, :, :, -1:]
    x = x[:, :, :, :-1]

    HRESIZE = 120  # Hack for now.
    WRESIZE = 160  # Hack for now.
    mask = tf.image.resize(mask, (HRESIZE, WRESIZE))
    x = tf.image.resize(x, (HRESIZE, WRESIZE))

    if self.config.data_format == 'NCHW':
      h = self.backbone(tf.transpose(x, [0, 3, 1, 2]))
      # Resize mask
      H = tf.shape(h)[2]
      W = tf.shape(h)[3]
    else:
      h = self.backbone(x)
      # Resize mask
      H = tf.shape(h)[1]
      W = tf.shape(h)[2]

    # ROI into 1 vector.
    mask_small = tf.image.resize(mask, [H, W])

    if self.config.data_format == 'NCHW':
      mask_small = mask_small[:, None, :, :, 0]

    obj = self._mask_pool(h, mask_small)
    context = self._global_avg_pool(h)

    if self._backbone.config.add_context_dropout and is_training:
      context = tf.nn.dropout(context, self.config.context_dropout_rate)

    # Concatenate context feature and object feature.
    f = tf.concat([obj, context], axis=-1)
    return f

  @property
  def backbone(self):
    return self._backbone
