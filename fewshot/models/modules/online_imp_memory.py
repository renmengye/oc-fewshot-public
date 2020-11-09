"""Online mixture memory that performs online clustering.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.example_memory import ExampleMemory
from fewshot.models.registry import RegisterModule

INF = 1e6


@RegisterModule("online_imp_memory")
@RegisterModule("online_mixture_memory")  # Legacy name
class OnlineIMPMemory(ExampleMemory):

  def forward_one(self,
                  x,
                  y,
                  t,
                  cmean,
                  clabel,
                  cusage,
                  is_training=tf.constant(True)):
    y_ = self.retrieve(x, t, cmean, clabel, cusage, is_training=is_training)
    # klogits, remain, new = self._infer_one(x, cmean, clabel, cusage, y=y)
    klogits, new = self._infer_one(x, cmean, clabel, cusage, y=y)
    new_id = tf.reduce_sum(tf.cast(tf.greater(cusage, 0), tf.int64),
                           [1])  # [B]
    kidx = tf.where(tf.less(new, 0.0), tf.argmax(klogits, axis=1), new_id)
    cmean, clabel, cusage = self.store(x, kidx, y, t, cmean, clabel, cusage)
    return y_, (cmean, clabel, cusage)

  def retrieve(self,
               x,
               t,
               cmean,
               clabel,
               cusage,
               is_training=tf.constant(True)):
    # clogits, remain, new = self.infer(x, t, cmean, clabel, cusage)
    clogits, new = self.infer(x, t, cmean, clabel, cusage)
    new_ = tf.reshape(new, [-1, 1])
    pad = tf.zeros_like(clogits)[:, :-1] - INF
    # TODO use scatter_nd to assign unknown ID.
    logits_unk = tf.concat([pad, new_], axis=1)
    logits = tf.maximum(clogits, logits_unk)
    return logits

  def infer(self, x, t, cmean, clabel, cusage):
    """Infer cluster ID. Either goes into one of the existing cluster
    or become a new cluster. This procedure is for prediction purpose.

    Args:
      x: Input. [B, D]
      cmean: Cluster centers. [B, K, D]
      clabel: Cluster labels. [B, K]
      cusage: Usage binary vector for the cluster. [B, K]

    Returns:
      logits: Cluster logits. [B, M]
      new_prob: New cluster probability. [B]
    """
    # logits, remain, new = self._infer_one(x, cmean, clabel, cusage)
    logits, new = self._infer_one(x, cmean, clabel, cusage)
    kprob = tf.nn.softmax(logits)  # [B, K]
    clabel_onehot = tf.one_hot(clabel, self.unknown_id + 1)  # [B, K', C]
    # [B, K, 1] * [B, K, C] = [B, C]
    cprob = tf.reduce_sum(tf.expand_dims(kprob, -1) * clabel_onehot, [1])
    cprob = tf.maximum(cprob, 1e-6)  # Delta.
    return tf.math.log(cprob), new

  def get_initial_state(self, bsize):
    """Initial state for the RNN."""
    M = self.max_items
    dim = self.dim

    # Cluster storage.
    cmean = tf.zeros([bsize, M, dim], dtype=self.dtype)
    clabel = tf.zeros([bsize, M], dtype=tf.int64)

    # Number of examples per cluster.
    cusage = tf.zeros([bsize, M], dtype=self.dtype)
    return cmean, clabel, cusage

  def _infer_one(self, x, cmean, clabel, cusage, y=None, verbose=False):
    """Infers one example.

    Args:
      x: Input. [B, D]
      cmean: Cluster centers. [B, K, D]
      clabel: Cluster labels. [B, K]
      cusage: Usage binary vector for the cluster. [B, K]

    Returns:
      logits: Cluster logits. [B, M]
      remain: Old cluster logit. [B]
    """
    # verbose = y is not None
    # verbose = False
    # Whether a cluster is used.
    cusage_flag = tf.greater(cusage, 0)  # [B, K]

    # Returns cluster ID and label.
    x_ = tf.expand_dims(x, 1)  # [B, 1, D]
    pdist = tf.squeeze(self.compute_euclidean_dist_sq(x_, cmean), 1)  # [B, K]
    pdist += tf.where(cusage_flag, 0.0, INF)

    if y is not None:
      y_ = tf.expand_dims(tf.cast(y, clabel.dtype), -1)  # [B]
      rel_flag = tf.logical_or(
          tf.equal(clabel, y_), tf.equal(clabel, self.unknown_id))
      pdist += tf.where(rel_flag, 0.0, INF)

    # Variance parameter.
    labeled_cluster = clabel < self.unknown_id
    sigma = tf.where(labeled_cluster, self.sigma_l, self.sigma_u)

    # Need to consider labeled case here.
    min_dist = tf.reduce_min(pdist, [-1])  # [B]
    # remain = (self._beta - min_dist) / self._gamma  # [B]
    new = (min_dist - self._beta) / self._gamma  # [B]
    pdist = pdist / (2.0 * sigma**2)
    return -pdist, new

  def store(self, x, kidx, y, t, cmean, clabel, cusage):
    """Stores a new example.

    Args:
      x: Input. [B, ...].
      kidx: Cluster Idx. [B]
      y: Label. [B]
      t: Int. Timestep.
      cmean: [B, M, D].
      clabel: [B, M].
      cusage: [B, M].
    """
    # Push into the example storage.
    bidx = tf.range(x.shape[0], dtype=tf.int64)  # [B]
    bkidx = tf.stack([bidx, kidx], axis=-1)  # [B, 2]
    # cusage_ = tf.cast(tf.expand_dims(cusage, -1), self.dtype)  # [B, M, 1]

    cmean_cur = tf.gather_nd(cmean, bkidx)  # [B, D]
    count = tf.gather_nd(cusage, bkidx)  # [B]
    count_ = tf.expand_dims(count, -1)  # [B]
    cmean_update = cmean_cur * count_ / (count_ + 1.0) + x / (count_ + 1.0)
    cmean_new = tf.tensor_scatter_nd_update(cmean, bkidx, cmean_update)

    cusage_update = count + 1
    cusage_new = tf.tensor_scatter_nd_update(cusage, bkidx, cusage_update)

    clabel_cur = tf.gather_nd(clabel, bkidx)  # [B]
    clabel_cur = tf.where(tf.greater(count, 0), clabel_cur, self.unknown_id)
    # Prefer labeled vs. unlabeled.
    clabel_upd = tf.minimum(clabel_cur, tf.cast(y, clabel_cur.dtype))
    clabel_new = tf.tensor_scatter_nd_update(clabel, bkidx, clabel_upd)
    return cmean_new, clabel_new, cusage_new
