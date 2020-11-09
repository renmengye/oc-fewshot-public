"""A semi-supervised proto memory that assigns soft probabilities to known
entities. Note: this is not used. The difference between this one and v2 is
that this one adds the soft probability, whereas v2 directly uses the argmax
of the prediction head.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.proto_memory import ProtoMemory
LOGINF = 1e6


# Warning: not used.
class SemiSupervisedProtoMemory(ProtoMemory):

  def __init__(self,
               name,
               dim,
               max_classes=20,
               fix_unknown=False,
               unknown_id=None,
               similarity="euclidean",
               temp_init=10.0,
               dtype=tf.float32):
    super(SemiSupervisedProtoMemory, self).__init__(
        name,
        dim,
        max_classes=max_classes,
        fix_unknown=fix_unknown,
        unknown_id=unknown_id,
        similarity=similarity,
        temp_init=temp_init,
        dtype=dtype)
    assert fix_unknown, 'Not supported'

  def forward_one(self,
                  x,
                  y,
                  t,
                  storage,
                  count,
                  add_new=tf.constant(True),
                  is_training=tf.constant(True),
                  **kwargs):
    y_ = self.retrieve(
        x, storage, count, t, add_new=add_new, is_training=is_training)
    y_soft = tf.nn.softmax(y_)
    storage, count = self.store(x, y, storage, count, y_soft=y_soft)
    return y_, (storage, count)

  def store(self, x, y, storage, count, y_soft=None):
    """Store a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
      storage: Storage. [B, K, D].
      count: Count. [B, K].
      y_soft: Soft label. [B, K].
    """
    # TODO modify y and try hard ID here.
    bidx = tf.range(y.shape[0], dtype=y.dtype)  # [B]
    idx = tf.stack([bidx, y], axis=1)  # [B, 2]
    curr_val = tf.gather_nd(storage, idx)  # [B, D]
    curr_cnt = tf.expand_dims(tf.gather_nd(count, idx), 1)  # [B, 1]
    new_val = (curr_val * curr_cnt + x) / (curr_cnt + 1.0)  # [B, D]
    update = new_val - curr_val
    # Do not increment for unknowns.
    if self._fix_unknown:
      inc = 1.0 - tf.cast(tf.equal(y, self._unknown_id), self.dtype)  # [B]
      storage_new = storage + tf.scatter_nd(
          idx, update * tf.expand_dims(inc, 1), storage.shape)  # [B, N, D]
    else:
      inc = tf.ones([y.shape[0]], dtype=self.dtype)
      storage_new = storage + tf.scatter_nd(idx, update, storage.shape)
    count_new = count + tf.scatter_nd(idx, inc, count.shape)  # [B]

    if y_soft is not None:
      mask = tf.cast(tf.greater(count, 0), y_soft.dtype)  # [B, K]
      y_soft_mask = y_soft * mask  # [B, K]

      # TODO maybe consider doing argmax here for better performance.
      unk_mask = tf.cast(tf.equal(y, self._unknown_id), y_soft.dtype)  # [B]
      unk_mask = tf.reshape(unk_mask, [-1, 1, 1])  # [B, 1, 1]

      # [B, 1, D] * [B, K, 1] * [B, 1, 1] = [B, K, D]
      unk_update = tf.expand_dims(x, 1) * tf.expand_dims(y_soft_mask,
                                                         -1) * unk_mask
      storage_new = storage_new * tf.expand_dims(count_new, -1) + unk_update
      count_new += y_soft_mask * unk_mask[:, :, 0]
      storage_new /= tf.expand_dims(
          count_new + tf.cast(tf.equal(count_new, 0), self.dtype), -1)
    return storage_new, count_new
