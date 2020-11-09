"""Online meta-learning module.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope
from fewshot.models.modules.mlp import MLP, CosineLastMLP
# from fewshot.models.modules.nnlib import CosineLinear
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModule("oml")
class OML(ContainerModule):

  def __init__(self, name, config, dtype=tf.float32):
    super(OML, self).__init__(dtype=dtype)
    self._name = name
    self._config = config

    self._oml_mlp = self.build_net()
    self._fast_weights = self._oml_mlp.weights()
    self._fast_weights_keys = [
        v.name.split(":")[0] for v in self._fast_weights
    ]
    log.info('Fast weights {}'.format(self._fast_weights_keys))
    # # self._bias_init =
    # with variable_scope(name):
    #   # learn initial bias values for the last layer.
    #   self._bias_init = self._get_variable(
    #       'bias_init', lambda: tf.zeros([1]) - 1.0, dtype=self.dtype)
    if config.learn_weight_init:
      self._weight_init = self.build_weight_init()

    if config.unknown_logits == "radii":
      self._beta = self._get_variable(
          "beta", lambda: tf.zeros([1]) + 10.0, dtype=self.dtype)
      self._gamma = self._get_variable(
          "gamma", lambda: tf.ones([1]), dtype=self.dtype)

      if config.semisupervised:
        self._beta2 = self._get_variable(
            "beta2", lambda: tf.zeros([1]) + 10.0, dtype=self.dtype)
        self._gamma2 = self._get_variable(
            "gamma2", lambda: tf.ones([1]), dtype=self.dtype)

  def build_net(self, wdict=None):
    with variable_scope(self._name):
      K = self.config.num_classes
      if self.config.inner_loop_loss == "mix":
        K += 1
      if self.config.cosine_classifier:
        # mlp = CosineLinear(
        #     "mlp",
        #     # list(self.config.num_filters) + [K],
        #     self.config.num_filters[0],
        #     K,
        #     temp=10.0,
        #     wdict=wdict)
        mlp = CosineLastMLP(
            "mlp",
            list(self.config.num_filters) + [K],
            # self.config.num_filters[0], K,
            temp=10.0,
            wdict=wdict)
      else:
        mlp = MLP(
            "mlp",
            list(self.config.num_filters) + [K],
            wdict=wdict,
            add_bias=self.config.classifier_bias)
    return mlp

  # Note that python function allows maximum 255 arguments for versions before
  # 3.7
  def forward(self, x, y, *states, **kwargs):
    B = x.shape[0]
    K = self.config.num_classes
    out = tf.TensorArray(self.dtype, size=B)  # Output prediction logits.
    unk = tf.TensorArray(self.dtype, size=B)  # Unknown prediction logits.
    class_count = states[-1]
    states = states[:-1]
    states_new = []
    for i in range(len(states)):
      states_new.append(tf.TensorArray(self.dtype, size=B))

    a = self.config.inner_lr

    # Run one step gradient descent for each examples in the mini-batch.
    for b in tf.range(B):
      fast_weights = [s[b] for s in states]
      x_ = x[b:b + 1]  # [1, D]
      y_ = y[b:b + 1]  # [1, D]
      k_ = class_count[b:b + 1]  # [1]
      pred_out_ = tf.zeros([1, self.config.num_classes], dtype=self.dtype)
      pred_unk_ = tf.zeros([1], dtype=self.dtype)
      loss_ = tf.zeros([1], dtype=self.dtype)
      end = tf.zeros([], dtype=tf.int64)
      unk2_ = tf.zeros([1], dtype=self.dtype)
      for tau in tf.range(self.config.repeat_steps):
        with tf.GradientTape() as tape_:

          if self.config.learn_weight_init:
            # We need to add the learned weight initialization.
            fast_weights = [
                v + v0 for v, v0 in zip(fast_weights, self._weight_init)
            ]

          for v in fast_weights:
            tape_.watch(v)
          wdict_ = dict(zip(self.fast_weights_keys, fast_weights))
          mlp_ = self.build_net(wdict=wdict_)
          out_ = mlp_(x_)

          if self.config.unknown_output_type == "softmax":
            if self.config.select_active_classes:
              # Select the active classes.
              end = tf.maximum(tf.minimum(k_[0] + 1, K), 1) + 1
              if end < K + 1:
                out_smax_ = tf.concat(
                    [tf.nn.softmax(out_[:, :end]),
                     tf.zeros([1, K - end])],
                    axis=1)
              else:
                out_smax_ = tf.nn.softmax(out_)
            else:
              out_smax_ = tf.nn.softmax(out_)  # [1, K]
          elif self.config.unknown_output_type == "sigmoid":
            out_smax_ = tf.nn.sigmoid(out_)  # [1, K]
          else:
            raise ValueError("Unknown inner loop loss")

          # Get the unknown output.
          kmask = tf.cast(
              tf.greater(
                  tf.range(self.config.num_classes, dtype=tf.int64)[None, :],
                  k_), self.dtype)  # [1, K]

          kmask2 = tf.cast(
              tf.less_equal(
                  tf.range(self.config.num_classes, dtype=tf.int64)[None, :],
                  k_), self.dtype)  # [1, K]

          if self.config.unknown_logits == "sum":
            unk_ = tf.reduce_sum(out_smax_ * kmask, [-1])  # [1]
            unk_ = tf.minimum(tf.maximum(unk_, 1e-5), 1 - 1e-3)
            unk_ = -tf.math.log(1 / unk_ - 1)
            unk2_ = unk_
          elif self.config.unknown_logits == "max":
            unk_ = 1.0 - tf.reduce_max(out_smax_ * kmask2, [-1])  # [1]
            unk_ = tf.minimum(tf.maximum(unk_, 1e-5), 1 - 1e-3)
            unk_ = -tf.math.log(1 / unk_ - 1)
            unk2_ = unk_
          elif self.config.unknown_logits == "radii":
            max_known_ = tf.reduce_max(out_ * kmask2, [-1])  # [1]
            unk_ = (-max_known_ + self._beta) / self._gamma
            unk2_ = (-max_known_ + self._beta2) / self._gamma2
          elif self.config.unknown_logits == "last":
            unk_ = -out_[:, -1]  # [1]
            unk2_ = unk_
          else:
            raise ValueError()
          unk2_.set_shape([1])

          # Make prediction on the first timestep (before seeing the labels).
          if tf.equal(tau, 0):
            pred_out_ = out_
            if self.config.inner_loop_loss == "mix":
              pred_out_ = out_[:, :-1]
            else:
              pred_out_ = out_
            pred_unk_ = unk_
            pred_out_.set_shape([1, self.config.num_classes])
            pred_unk_.set_shape([1])

          # Inner loss is trained on only known labels.
          # TODO try semisupervised here.
          flag_ = tf.where(tf.equal(y_, K), 0.0, 1.0)
          y_ = tf.where(tf.equal(y_, K), tf.zeros_like(y_), y_)
          y_onehot_ = tf.one_hot(y_, K)

          if self.config.select_active_classes:
            # Select the active classes.
            if self.config.inner_loop_loss == "softmax":
              end = tf.maximum(tf.minimum(k_[0] + 1, K), 1) + 1
            elif self.config.inner_loop_loss == "sigmoid":
              end = tf.minimum(k_[0] + 1, K) + 1
            out_ = out_[:, :end]
            y_onehot_ = y_onehot_[:, :end]

          if self.config.inner_loop_loss == "softmax":
            loss_ = tf.nn.softmax_cross_entropy_with_logits(
                logits=out_, labels=y_onehot_) * flag_
            if self.config.semisupervised:
              pseudo_label_ = tf.one_hot(tf.argmax(out_, axis=1), K)  # [B, K]
              ssl_flag_ = (1.0 - flag_) * unk2_  # [B, K]
              loss_ += tf.nn.softmax_cross_entropy_with_logits(
                  logits=out_, labels=pseudo_label_) * ssl_flag_
          elif self.config.inner_loop_loss == "sigmoid":
            # Only update the current label. See if it works.
            loss_ = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=out_[:, y_[0]], labels=[1.0]) * flag_)
            assert not self.config.semisupervised
          elif self.config.inner_loop_loss == "mix":
            assert not self.config.select_active_classes
            loss1_ = tf.nn.softmax_cross_entropy_with_logits(
                logits=out_[:, :-1], labels=y_onehot_)
            loss2_ = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=out_[:, -1], labels=[1.0])
            loss_ = (loss1_ + loss2_) * flag_
          else:
            raise ValueError("Unknown inner loop loss")

        # TODO: Add check for semisupervised version.
        grad_ = tape_.gradient(loss_, fast_weights)
        fast_weights = [
            fw_i - a * g_i for fw_i, g_i, in zip(fast_weights, grad_)
        ]

      # Inner loop gradient descent.
      if self.config.learn_weight_init:
        states_new = [
            s.write(b, fw_i - w0_i) for s, fw_i, w0_i in zip(
                states_new, fast_weights, self._weight_init)
        ]
      else:
        states_new = [
            s.write(b, fw_i) for s, fw_i in zip(states_new, fast_weights)
        ]

      # Update class count.
      class_inc = tf.tensor_scatter_nd_update(-tf.ones([B, 1], dtype=y_.dtype),
                                              [[b, 0]], [y_[0]])
      class_inc = tf.cast(class_inc[:, 0], class_count.dtype)
      class_count = tf.maximum(class_count, class_inc)

      # Write the output.
      out = out.write(b, pred_out_[0])
      unk = unk.write(b, pred_unk_[0])

    # assert False
    out = out.stack()  # [B, K]
    unk = unk.stack()  # [B]

    # tf.print('unk', unk, summarize=16)
    states_new = [s.stack() for s in states_new]
    shapelist = [[int(s) for s in v.shape] for v in self.fast_weights]
    for s, shape in zip(states_new, shapelist):
      s.set_shape([B] + shape)
    return out, unk, (*states_new, class_count)

  def get_initial_state(self, bsize):
    """Initial state of the fast weights, with a mini batch.

    Args:
      bsize: Int. Batch size.

    Returns:
      fast_weights: List of fast weights, the first dimension is the batch
        size.
    """
    state_list = []
    varlist = self.fast_weights
    log.info('varlist {}'.format(varlist))
    L = len(varlist)
    shapelist = [[int(s) for s in v.shape] for v in varlist]
    fast_weights = [None] * L
    for i in range(L):
      n = shapelist[i][-1]
      # fast_weights[i] = tf.zeros([bsize] + shapelist[i], dtype=self.dtype)
      key = self.fast_weights_keys[i]

      if L == 2 or self.config.learn_weight_init:
        fast_weights[i] = tf.zeros([bsize] + shapelist[i], dtype=self.dtype)
      else:
        if key.endswith('b'):
          fast_weights[i] = tf.zeros([bsize] + shapelist[i], dtype=self.dtype)
          log.info('weights {} zero init'.format(key))
        else:
          fast_weights[i] = tf.random.truncated_normal(
              [bsize] + shapelist[i], mean=0.0, stddev=0.01)
          log.info('weights {} normal init'.format(key))
    class_count = tf.zeros([bsize], dtype=tf.int64) - 1
    return (*fast_weights, class_count)

  def build_weight_init(self):
    """Initial state of the fast weights, with a mini batch.

    Args:
      bsize: Int. Batch size.

    Returns:
      fast_weights: List of fast weights, the first dimension is the batch
        size.
    """
    state_list = []
    varlist = self.fast_weights
    log.info('varlist {}'.format(varlist))
    L = len(varlist)
    shapelist = [[int(s) for s in v.shape] for v in varlist]
    fast_weights = [None] * L
    weight_init_list = []

    with variable_scope(self._name):
      for i in range(L):
        key = self.fast_weights_keys[i]
        if key.endswith('b'):

          def binit():
            return tf.zeros(shapelist[i])

          b = self._get_variable(
              'layer_{}/b_init'.format(i // 2), binit, dtype=self._dtype)
          weight_init_list.append(b)
        else:

          def winit():
            return tf.random.truncated_normal(
                shapelist[i], mean=0.0, stddev=0.01)

          idx = i // 2 if self.config.classifier_bias else i
          w = self._get_variable(
              'layer_{}/w_init'.format(idx), winit, dtype=self._dtype)
          weight_init_list.append(w)
    return weight_init_list

  @property
  def fast_weights_keys(self):
    return self._fast_weights_keys

  @property
  def fast_weights(self):
    return self._fast_weights

  @property
  def config(self):
    return self._config
