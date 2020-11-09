"""Prototypical memory modules.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ProtoMemory(ContainerModule):
  """A module that stores prototypes."""

  def __init__(self,
               name,
               dim,
               max_classes=20,
               fix_unknown=False,
               unknown_id=None,
               similarity="euclidean",
               temp_init=10.0,
               dtype=tf.float32):
    super(ProtoMemory, self).__init__(dtype=dtype)
    self._max_classes = max_classes
    self._fix_unknown = fix_unknown
    self._unknown_id = unknown_id
    self._similarity = similarity
    self._dim = dim
    if fix_unknown:
      log.info('Fixing unknown id')
      assert unknown_id is not None, 'Need to provide unknown ID'

    if similarity in ["cosine", "poincare"]:
      with variable_scope(name):
        self._temp = self._get_variable("temp",
                                        self._get_constant_init([], temp_init))
        # self._temp = self._get_variable(
        #     "temp", self._get_constant_init([], temp_init), trainable=False)

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
    storage, count = self.store(x, y, storage, count)
    return y_, (storage, count)

  def compute_logits(self, x, prototypes):
    if self.similarity == "euclidean":
      dist = tf.reduce_sum(tf.square(x - prototypes), [-1])  # [B, K+1]
      return -dist
    elif self.similarity == "cosine":
      eps = 1e-5
      p = prototypes
      x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), [-1], keepdims=True))
      p_norm = tf.sqrt(tf.reduce_sum(tf.square(p), [-1], keepdims=True))
      # if tf.less(tf.reduce_min(x_norm), eps):
      #   tf.print('x norm too small', tf.reduce_min(x_norm))
      # if tf.less(tf.reduce_min(p_norm), eps):
      #   tf.print('p norm too small', tf.reduce_min(p_norm))

      # tf.print(x[0],)
      x_ = x / (x_norm + eps)
      p_ = p / (p_norm + eps)
      x_dot_p = tf.matmul(p_, tf.transpose(x_, [0, 2, 1]))[:, :, 0]  # [B, K]
      # tf.print()
      return x_dot_p * self._temp
    elif self.similarity == "poincare":
      # Prototypes are stored in Klein space.
      c = 0.04
      x = self._expmap0(x, c)
      p_k = prototypes
      p_d = p_k / (1 + tf.sqrt(
          1 - c * tf.reduce_sum(tf.square(p_k), [-1], keepdims=True)))
      dist = self._dist_matrix2(x, p_d, c)[:, 0, :]
      return -dist * self._temp

  def _expmap0(self, u, c):
    sqrt_c = c**0.5
    u_norm = tf.maximum(tf.norm(u, axis=-1, keepdims=True), 1e-5)
    gamma_1 = tf.math.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1

  def _mobius_addition_batch2(self, x, y, c):
    """
      x: N x B x D
      y: N x C x D
    """
    xy = tf.linalg.matmul(x, tf.transpose(y, [0, 2, 1]))  # N x B x C
    x2 = tf.reduce_sum(tf.square(x), [-1], keepdims=True)  # N x B x 1
    y2 = tf.reduce_sum(tf.square(y), [-1], keepdims=True)  # N x C x 1
    num = (1 + 2 * c * xy + c * tf.transpose(y2, [0, 2, 1]))  # N x B x C
    # N x B x C x 1 * N x B x 1 x D = N x B x C x D
    num = num[:, :, :, None]
    num = num * x[:, :, None, :]
    num = num + (1 - c * x2)[:, :, :, None] * y[:, None, :, :]  # N x B x C x D
    denom_part1 = 1 + 2 * c * xy  # N x B x C
    denom_part2 = c**2 * x2 * tf.transpose(y2, [0, 2, 1])  # N x B x C
    denom = denom_part1 + denom_part2
    res = num / (denom[:, :, :, None] + 1e-5)  # N x B x C x D
    return res

  def _dist_matrix2(self, x, y, c):
    """
      x: N x B x D
      y: N x C x D
      return: N x B x C
    """
    sqrt_c = c**0.5
    return 2 / sqrt_c * tf.math.atanh(
        sqrt_c * tf.norm(self._mobius_addition_batch2(-x, y, c=c), axis=-1))

  def _mobius_addition_batch(self, x, y, c):
    """
      x: B x D
      y: C x D
    """
    xy = tf.matmul(x, y, transpose_b=True)  # B x C
    x2 = tf.reduce_sum(tf.square(x), [-1], keepdims=True)  # B x 1
    y2 = tf.reduce_sum(tf.square(y), [-1], keepdims=True)  # C x 1
    num = (1 + 2 * c * xy + c * tf.transpose(y2))  # B x C
    num = num[:, :, None] * x[:, None, :]  # B x C x 1 * B x 1 x D = B x C x D
    num = num + (1 - c * x2)[:, :, None] * y[None, :, :]  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c**2 * x2 * tf.transpose(y2)
    denom = denom_part1 + denom_part2
    res = num / (denom[:, :, None] + 1e-5)
    return res

  def _dist_matrix(self, x, y, c):
    """
      x: B x D
      y: C x D
    """
    sqrt_c = c**0.5
    return 2 / sqrt_c * tf.math.atanh(
        sqrt_c * tf.norm(self._mobius_addition_batch(-x, y, c=c), axis=-1))

  def store(self, x, y, storage, count):
    """Store a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
    """
    bidx = tf.range(tf.shape(y)[0])  # [B]
    y = tf.cast(y, bidx.dtype)
    idx = tf.stack([bidx, y], axis=1)  # [B, 2]
    curr_val = tf.gather_nd(storage, idx)  # [B, D]
    curr_cnt = tf.expand_dims(tf.gather_nd(count, idx), 1)  # [B, 1]

    if self.similarity in ["poincare"]:
      c = 0.04
      x = self._expmap0(x, c)

      xk = 2 * x
      xk = xk / (1 + c * tf.reduce_sum(tf.square(x), [-1], keepdims=True))
      gamma = 1 / tf.sqrt(
          1 - c * tf.reduce_sum(tf.square(xk), [-1], keepdims=True))  # [B, 1]
      new_val = (curr_val * curr_cnt + xk * gamma) / (curr_cnt + gamma)
    else:
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

    if self.similarity in ["poincare"]:
      count_new = count + tf.scatter_nd(idx, inc * gamma[:, 0], count.shape)
    else:
      count_new = count + tf.scatter_nd(idx, inc, count.shape)
    return storage_new, count_new

  def get_initial_state(self, bsize):
    """Get initial states."""
    K = self.max_classes
    if self._fix_unknown:
      K += 1
    storage = tf.zeros([bsize, K, self._dim], dtype=self.dtype)
    count = tf.zeros([bsize, K], dtype=self.dtype)
    return storage, count

  @property
  def max_classes(self):
    """Maximum number of classes."""
    return self._max_classes

  @property
  def unknown_id(self):
    """Unknown ID."""
    return self._unknown_id

  @property
  def similarity(self):
    """Similarity function."""
    return self._similarity

  @property
  def dim(self):
    """Storage dimensionality."""
    return self._dim
