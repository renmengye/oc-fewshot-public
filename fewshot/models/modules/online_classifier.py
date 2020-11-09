"""Online one vs. rest SVM -- an online version of the OvA SVM.

This version is built on top of sklearn and numpy, and therefore does not
support graph mode.

Just linear SVM for now. Maybe adapt to others later.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.modules.example_memory import ExampleMemory
from fewshot.models.registry import RegisterModule
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

INF = 1e6


class OnlineClassifier(ExampleMemory):

  def infer(self, x, t, storage, label, clf_list):
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
    x_numpy = x.numpy()
    y_all = [[] for b in range(B)]
    for b in range(B):
      clf = clf_list[b]
      y_, unk_ = self.predict(clf, x)
      rest = np.zeros([K + 1 - len(y_)]) - INF
      y_all[b] = np.concatenate([y_, rest])
    y_all = np.array(y_all)  # [B, K+1]
    logits = tf.constant(y_all, dtype=tf.float32)
    y_max = tf.reduce_max(logits, [-1])  # [B]
    new = (self._beta - y_max) / self._gamma
    return logits, new

  def new_classifier(self, x, y):
    """Train a new classifier."""
    raise NotImplementedError()

  def predict(self, clf, x):
    """Interpret classifier outputs."""
    raise NotImplementedError()

  def store(self, x, y, t, storage, label, clf_list):
    """Stores a new example.

    Args:
      x: Input. [B, ...].
      y: Label. [B].
      storage: Example storage. [B, M, ...].
      label: Example label. [B, M].
      t: Int. Timestep.
    """
    storage_new, label_new = super(OnlineClassifier, self).store(
        x, y, t, storage, label)
    B = int(x.shape[0])
    T = int(storage.shape[1])
    lbl_edit = tf.where(tf.less_equal(tf.range(T), t), label_new, -1)

    for b in range(B):
      lbl = lbl_edit[b, :min(t + 2, T + 1)].numpy()
      inp = storage_new[b, :min(t + 2, T + 1)].numpy()
      clf_list[b] = self.new_classifier(inp, lbl)
    return storage_new, label_new, clf_list

  def get_initial_state(self, bsize):
    """Initial state for the RNN."""
    M = self.max_items
    dim = self.dim

    # Cluster storage.
    storage = tf.zeros([bsize, M, dim], dtype=self.dtype)
    label = tf.zeros([bsize, M], dtype=tf.int64)
    clf_list = [[] for b in range(bsize)]
    return storage, label, clf_list


@RegisterModule("online_ovrsvm")
class OnlineOVRSVM(OnlineClassifier):

  def new_classifier(self, x, y):
    svm = LinearSVC(C=.1, class_weight='balanced', max_iter=10000)
    svm.fit(x, y)
    return svm

  def predict(self, clf, x):
    y_ = clf.decision_function(x)  # [K]
    if y_.size > 1:
      unk_ = y_[0, 0]
      y_ = y_[0, 1:]
    else:
      unk_ = -y_
    return y_, unk_


@RegisterModule("online_lr")
class OnlineLR(OnlineClassifier):

  def new_classifier(self, x, y):
    # svm = LinearSVC(C=.1, class_weight='balanced', max_iter=10000)
    # lr = LogisticRegression(C=1.0, max_iter=10000)  # 83.306
    # lr = LogisticRegression(
    #     C=1.0, class_weight='balanced', max_iter=10000)  # 91.276
    # lr = LogisticRegression(
    #     C=0.1, class_weight='balanced', max_iter=10000)  # 82.262
    # lr = LogisticRegression(
    #     C=10.0, class_weight='balanced', max_iter=10000)  # 91.730
    # lr = LogisticRegression(
    #     C=100.0, class_weight='balanced', max_iter=10000)  # 91.541
    # lr = LogisticRegression(
    #     C=1000.0, class_weight='balanced', max_iter=10000)  # 91.327
    # lr = LogisticRegression(C=10.0, max_iter=10000)  # 88.971
    # lr = LogisticRegression(C=100.0, max_iter=10000)  # 89.916
    lr = LogisticRegression(C=1000.0, max_iter=10000)  # 89.916

    lr.fit(x, y)
    return lr

  def predict(self, clf, x):
    y_ = clf.predict_log_proba(x)
    unk_ = y_[0, 0]
    y_ = y_[0, 1:]
    return y_, unk_
