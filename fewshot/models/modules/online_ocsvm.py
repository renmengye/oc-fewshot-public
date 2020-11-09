"""Online one-class SVM -- an online version of the OC-SVM.

This version is built on top of sklearn and numpy, and therefore does not
support graph mode.

TODO: to make it real online. Currently I am retraining a new classifier
every timestep. This can be potentially be made faster by introducing online
update, as described by:

- Gert Cauwenberghs, Tomaso A. Poggio.
Incremental and Decremental Support Vector Machine Learning.
NIPS 2000: 409-415.

- Pavel Laskov, Christian Gehl, Stefan Kruger, Klaus-Robert Muller.
Incremental Support Vector Learning: Analysis, Implementation and Applications.
JMLR. 7: 1909-1936 (2006).

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.modules.example_memory import ExampleMemory
from fewshot.models.registry import RegisterModule
from sklearn.svm import OneClassSVM

INF = 1e6


@RegisterModule("online_ocsvm")
class OnlineOCSVM(ExampleMemory):

  def infer(self, x, t, storage, label, svm_list):
    """Infer cluster ID. Either goes into one of the existing cluster
    or become a new cluster. This procedure is for prediction purpose.

    Args:
      x: Input. [B, D]
      t: Timestep. Int.
      storage: Example storage. [B, M, D].
      label: Example label. [B, M]
      svm_list: List of previously learned SVMs.

    Returns:
      logits: Cluster logits. [B, M]
      new_prob: New cluster probability. [B]
    """
    B = tf.shape(x)[0]
    K = self.max_classes
    if tf.equal(t, 0):
      return tf.zeros([B, K + 1],
                      dtype=self.dtype), tf.zeros([B], dtype=self.dtype) + INF
    storage_ = storage[:, :t, :]
    label_ = label[:, :t]
    # x_ = tf.expand_dims(x, 1)  # [B, 1, D]
    # Select current class.
    x_numpy = x.numpy()
    y_all = [[] for b in range(B)]
    for b in range(B):
      for clf in svm_list[b]:
        # clf = OneClassSVM(gamma='auto').fit(X)
        y_ = clf.score_samples(x_numpy)[0] - clf.offset_[0]
        y_all[b].append(y_)  # [K+1]
      # pad_b = np.zeros([K + 1 - len(y_all[b])], dtype=np.float32) - INF
      y_all[b].extend([-INF] * (K + 1 - len(svm_list[b])))

      # print('b', b, y_all[b])
    y_all = np.array(y_all)  # [B, K+1]
    logits = tf.constant(y_all, dtype=tf.float32)
    # if logits.shape[1] < K + 1:
    # pad = tf.zeros([B, K + 1 - logits.shape[1]], dtype=tf.float32) - INF
    # logits = tf.concat([logits, pad], axis=1)
    y_max = tf.reduce_max(logits, [-1])  # [B]
    print('y_max', y_max, self._beta, self._gamma)
    # print('SVM predict', y_all)
    new = (self._beta - y_max) / self._gamma
    return logits, new

  def store(self, x, y, t, storage, label, svm_list):
    """Stores a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
      storage: Example storage. [B, M, ...].
      label: Example label. [B, M].
      t: Int. Timestep.
    """
    storage_new, label_new = super(OnlineOCSVM, self).store(
        x, y, t, storage, label)
    B = int(x.shape[0])
    T = int(storage.shape[1])

    # TODO complete this part.
    for b in range(B):
      # Retrieve all examples belonging to class y.
      idx = tf.where(
          tf.logical_and(
              tf.equal(label_new[b], y[b]), tf.less_equal(tf.range(T),
                                                          t)))  # [M']
      storage_y = tf.gather_nd(storage_new[b], idx).numpy()  # [M', D]
      # label_y = tf.gather_nd(label, idx)

      svm_new = OneClassSVM(gamma="scale", nu=0.1, kernel='linear')
      svm_new.fit(storage_y)
      # print('y', y, 'SVM learn', svm_new.predict(storage_y),
      #       svm_new.score_samples(storage_y) - svm_new.offset_, 'num supp',
      #       svm_new.n_support_)
      print('y', y, 'count', storage_y.shape[0])
      print('overview', np.unique(label[0, :t].numpy(), return_counts=True))
      if t > 20:
        assert False
      y_ = y[b].numpy()
      if y_ == len(svm_list[b]):
        svm_list[b].append(svm_new)
      elif y_ < len(svm_list[b]):
        svm_list[b][y_] = svm_new
      else:
        assert False, 'Label cannot increment more than 1 at a time.'
    return storage_new, label_new, svm_list

  def get_initial_state(self, bsize):
    """Initial state for the RNN."""
    M = self.max_items
    dim = self.dim

    # Cluster storage.
    storage = tf.zeros([bsize, M, dim], dtype=self.dtype)
    label = tf.zeros([bsize, M], dtype=tf.int64)
    svm_list = [[] for b in range(bsize)]
    return storage, label, svm_list
