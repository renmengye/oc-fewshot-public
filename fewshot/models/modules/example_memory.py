"""Mixture memory that performs a clustering step.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.variable_context import variable_scope

INF = 1e6


class ExampleMemory(ContainerModule):

  def __init__(self,
               name,
               dim,
               max_items=80,
               max_classes=20,
               unknown_id=None,
               log_sigma_init=0.0,
               log_lambda_init=0.0,
               radius_init=10.0,
               similarity="euclidean",
               dtype=tf.float32):
    super(ExampleMemory, self).__init__(dtype=dtype)
    sigma_init = self._get_constant_init([], log_sigma_init)
    lbd_init = self._get_constant_init([], log_lambda_init)
    self._similarity = similarity
    self._dim = dim

    with variable_scope(name):
      self._log_sigma_u = self._get_variable("sigma_u", sigma_init)
      self._log_sigma_l = self._get_variable("sigma_l", sigma_init)
      self._beta = self._get_variable("beta",
                                      self._get_constant_init([], radius_init))
      self._gamma = self._get_variable("gamma", self._get_constant_init([],
                                                                        1.0))
    assert unknown_id is not None, 'Need to provide unknown ID'
    self._unknown_id = unknown_id
    self._max_classes = max_classes
    self._max_items = max_items

  def forward_one(self, x, y, t, *states, is_training=tf.constant(True)):
    """Forward one time step.

    Args:
      x: Float. Input. [B, ...]
      y: Int. Label. [B]
      mean: Float. Cluster centers. [B, K, ...]
      storage: Float. Example storage. [B, M, ...]
      label: Int. Cluster label. [B, K]
      usage: Int. Cluster usage. [B]
      t: Int. Timestep.
      is_training: Bool. Whether in training mode.
    """
    y = tf.cast(y, states[1].dtype)
    y_ = self.retrieve(x, t, *states, is_training=is_training)
    # print('t', t, 'y', y[0], 'y pred', y_[0])
    states = self.store(x, y, t, *states)
    return y_, states

  def compute_euclidean_dist_sq(self, a, b):
    """Computes Euclidean distance.

    Args:
      a: [B, K1, D] First input feature matrix.
      b: [B, K2, D] Second input feature matrix.

    Returns:
      dist: [B, K1, K2] Pairwise distance matrix.
    """
    a2 = tf.reduce_sum(tf.square(a), [-1], keepdims=True)  # [B, K1, 1]
    ab = tf.matmul(a, b, transpose_b=True)  # [B, K1, K2]
    b2 = tf.expand_dims(tf.reduce_sum(tf.square(b), [-1]), 1)  # [B, 1, K2]
    return a2 - 2 * ab + b2  # [B, K1, K2]

  def retrieve(self, x, t, *states, is_training=tf.constant(True)):
    """Retrieve the prediction.

    Args:
      x: Input. [B, ...].
      cmean: Cluster center. [B, K, ...].
      clabel: Cluster label. [B, K].
      cusage: Cluster allocation usage. [B].
      t: Time step.
      is_training:
    """
    # print('x', x.shape, 't', t, storage.shape, label.shape)
    # klogits, remain, new = self.infer(x, t, storage, label)  # [B, K]
    klogits, new = self.infer(x, t, *states)  # [B, K]
    # print('hey', klogits.shape, remain.shape, new.shape)
    new_ = tf.reshape(new, [-1, 1])
    pad = tf.zeros_like(klogits)[:, :-1] - INF

    # TODO use scatter_nd to assign unknown ID.
    logits_unk = tf.concat([pad, new_], axis=1)
    logits = tf.maximum(klogits, logits_unk)
    return logits

  def infer(self, x, t, storage, label):
    """Infer cluster ID. Either goes into one of the existing cluster
    or become a new cluster. This procedure is for prediction purpose.

    Args:
      x: Input. [B, D]

    Returns:
      logits: Cluster logits. [B, M]
      new_prob: New cluster probability. [B]
    """
    raise NotImplementedError()

  def store(self, x, y, t, storage, label):
    """Stores a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
      storage: Example storage. [B, M, ...].
      label: Example label. [B, M].
      t: Int. Timestep.
    """
    # Push into the example storage.
    bidx = tf.range(x.shape[0])  # [B]
    tidx = tf.ones([x.shape[0]], dtype=bidx.dtype) * t
    eidx = tf.stack([bidx, tidx], axis=1)

    storage_new = storage + tf.scatter_nd(eidx, x, storage.shape)
    label_new = label + tf.scatter_nd(eidx, y, label.shape)

    # No store for UNK
    cond = tf.equal(y, self.unknown_id)
    storage_new = tf.where(cond[:, None, None], storage, storage_new)
    label_new = tf.where(cond[:, None], label, label_new)

    return storage_new, label_new

  def get_initial_state(self, bsize):
    """Initial state for the RNN."""
    M = self.max_items
    dim = self.dim

    # Cluster storage.
    storage = tf.zeros([bsize, M, dim], dtype=self.dtype)
    label = tf.zeros([bsize, M], dtype=tf.int64)
    return storage, label

  @property
  def max_classes(self):
    """Maximum number of classes."""
    return self._max_classes

  @property
  def max_items(self):
    """Maximum number of example items."""
    return self._max_items

  @property
  def dim(self):
    """Dimension."""
    return self._dim

  @property
  def sigma_u(self):
    """Standard deviation for unlabeled cluster."""
    return tf.exp(self._log_sigma_u)

  @property
  def sigma_l(self):
    """Standard deviation for labeled cluster."""
    return tf.exp(self._log_sigma_l)

  @property
  def lambda_scale(self):
    """Scaling term for soft new cluster prediction."""
    return tf.exp(self._log_lambda_scale)

  @property
  def unknown_id(self):
    """ID reserved for the unknown class."""
    return self._unknown_id
