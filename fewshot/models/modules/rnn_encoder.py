"""ProtoNet memory plus a DNC module.

Use DNC module to
  1) encode the examples (with context);
  2) output unknown.

Use ProtoNet to
  1) decode the class;
  2) store class prototypes.

SSL predicts beta, gamma and temperature for y_soft.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.nnlib import Linear
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.mlp import MLP
from fewshot.models.modules.resmlp import ResMLP
from fewshot.utils.logger import get as get_logger

log = get_logger()


def swish(x):
  return tf.math.sigmoid(x) * x


@RegisterModule('rnn_encoder')
class RNNEncoder(ContainerModule):

  def __init__(self,
               name,
               rnn_memory,
               proto_memory,
               readout_type='linear',
               use_pred_beta_gamma=True,
               use_feature_fuse=True,
               use_feature_fuse_gate=True,
               use_feature_scaling=True,
               use_feature_memory_only=False,
               skip_unk_memory_update=False,
               use_ssl=True,
               use_ssl_beta_gamma_write=True,
               use_ssl_temp=True,
               dtype=tf.float32):
    super(RNNEncoder, self).__init__(dtype=dtype)
    self._rnn_memory = rnn_memory
    self._proto_memory = proto_memory

    # ------------- Feature Fusing Capability Ablation --------------
    self._use_pred_beta_gamma = use_pred_beta_gamma  # CHECK
    self._use_feature_fuse = use_feature_fuse  # CHECK
    self._use_feature_fuse_gate = use_feature_fuse_gate  # CHECK
    self._use_feature_scaling = use_feature_scaling  # CHECK
    self._use_feature_memory_only = use_feature_memory_only  # CHECK

    # ------------- SSL Capability Ablation --------------
    self._skip_unk_memory_update = skip_unk_memory_update  # CHECK
    self._use_ssl = use_ssl  # CHECK
    self._use_ssl_beta_gamma_write = use_ssl_beta_gamma_write  # CHECK
    self._use_ssl_temp = use_ssl_temp  # CHECK

    D_in = self._rnn_memory.memory_dim
    D = self._rnn_memory.in_dim
    self._dim = D

    # h        [D]
    # scale    [D]
    # temp     [1]
    # gamma2   [1]
    # beta2    [1]
    # gamma    [1]
    # beta     [1]
    # x_gate   [1]
    # h_gate   [1]
    bias_init = [
        tf.zeros(D),
        tf.zeros(D),
        tf.zeros([1]),
        tf.zeros([1]),
        tf.zeros([1]) + proto_memory._radius_init,
        tf.zeros([1]),
        tf.zeros([1]) + proto_memory._radius_init_write,
        tf.zeros([1]) + 1.0,
        tf.zeros([1]) - 1.0
    ]
    bias_init = tf.concat(bias_init, axis=0)

    D_out = bias_init.shape[-1]

    def b_init():
      return bias_init

    if readout_type == 'linear':
      log.info("Using linear readout")
      self._readout = Linear('readout', D_in, D_out, b_init=b_init)
    elif readout_type == 'mlp':
      log.info("Using MLP readout")
      self._readout = MLP(
          'readout_mlp', [D_in, D_out, D_out],
          bias_init=[None, bias_init],
          act_func=[tf.math.tanh])
    elif readout_type == 'resmlp':
      log.info("Using ResMLP readout")
      self._readout = ResMLP(
          'readout_mlp', [D_in, D_out, D_out, D_out],
          bias_init=[None, None, bias_init],
          act_func=[swish, swish, None])

  def forward(self, t, x, y, *states, store=False, ssl_store=None):
    D = self._dim
    rnn_states = states[:self._num_rnn_states]
    proto_states = states[self._num_rnn_states:-3]
    y_last = states[-3]
    h_last = states[-2]
    pred_last = states[-1]
    x_rnn = x
    rnn_out, rnn_states_new = self._rnn_memory.forward(x_rnn, *rnn_states)
    assert store or t == 0

    if self._skip_unk_memory_update:
      pass

      def expand(b, s):
        if len(s.shape) == 2:
          return b[:, None]
        elif len(s.shape) == 3:
          return b[:, None, None]
        elif len(s.shape) == 1:
          return b
        elif len(s.shape) == 0:
          return tf.constant(True)
        else:
          assert False

      rnn_states = [
          tf.where(
              expand(tf.equal(y, self.proto_memory.unknown_id), s), s, s_new)
          for s, s_new in zip(rnn_states, rnn_states_new)
      ]
    else:
      rnn_states = rnn_states_new

    readout = self.readout(rnn_out)
    h = readout[:, :D]
    scale = tf.math.softplus(readout[:, D:2 * D])
    h_gate = tf.math.sigmoid(readout[:, -1:])
    x_gate = tf.math.sigmoid(readout[:, -2:-1])
    beta = readout[:, -3]
    gamma = tf.nn.softplus(readout[:, -4] + 1.0)
    beta2 = readout[:, -5]
    gamma2 = tf.nn.softplus(readout[:, -6] + 1.0)
    temp = tf.nn.softplus(readout[:, -7:-6] + 1.0)
    count = proto_states[1]

    if ssl_store is None:
      ssl_store = tf.ones([x.shape[0]], dtype=tf.bool)

    if not self._use_ssl:
      ssl_store = tf.zeros([x.shape[0]], dtype=tf.bool)

    if store:
      pred_last_unk = tf.math.sigmoid(pred_last[:, -1:])
      pred_last_cls = tf.nn.softmax(pred_last[:, :-1])
      h_store = h_last
      y_soft = tf.concat(
          [pred_last_cls * (1.0 - pred_last_unk), pred_last_unk], axis=-1)
      proto_states_ssl = self._proto_memory.store(
          h_store, y_last, *proto_states, y_soft=y_soft)
      proto_states_nossl = self._proto_memory.store(h_store, y_last,
                                                    *proto_states)

      storage_new = tf.where(ssl_store[:, None, None], proto_states_ssl[0],
                             proto_states_nossl[0])
      count_new = tf.where(ssl_store[:, None], proto_states_ssl[1],
                           proto_states_nossl[1])
      storage_new.set_shape(
          [x.shape[0], self._proto_memory.max_classes + 1, D])
      count_new.set_shape([x.shape[0], self._proto_memory.max_classes + 1])
      proto_states = (storage_new, count_new)
      # print('count new', count_new)
      # print('storage new', storage_new)

    if self._use_feature_fuse:
      if self._use_feature_fuse_gate:
        x = x * x_gate + h * h_gate
      else:
        x = x + h
    else:
      pass

    if self._use_feature_memory_only:
      x = h

    storage, count = proto_states

    if self._use_feature_scaling:
      storage_scale = storage * scale[:, None, :]
      x_scale = x * scale
    else:
      storage_scale = storage
      x_scale = x

    if self._use_pred_beta_gamma:
      pass
    else:
      beta = None
      gamma = None
    y_cls = self.proto_memory.retrieve(
        x_scale, storage_scale, count, t, beta=beta, gamma=gamma, add_new=True)

    y_cls_idx = tf.argmax(y_cls[:, :-1], axis=-1)  # [B]
    B = tf.shape(x)[0]
    bidx = tf.range(B, dtype=y_cls_idx.dtype)  # [B]
    bidx = tf.stack([bidx, y_cls_idx], axis=-1)  # [B, 2]
    count_now = tf.gather_nd(count, bidx)  # [B]

    if self._use_pred_beta_gamma:
      if self._use_ssl_beta_gamma_write:
        pass
      else:
        beta2 = beta
        gamma2 = gamma
    else:
      if self._use_ssl_beta_gamma_write:
        beta2 = self.proto_memory._beta2
        gamma2 = self.proto_memory._gamma2
      else:
        beta2 = None
        gamma2 = None

    if self._use_ssl_temp:
      pass
    else:
      temp = None

    pred_last = self.proto_memory.retrieve(
        x_scale,
        storage_scale,
        count,
        t,
        beta=beta2,
        gamma=gamma2,
        temp=temp,
        add_new=True)

    h_last = x
    y_last = tf.cast(y, y_last.dtype)
    states = (*rnn_states, *proto_states, y_last, h_last, pred_last)
    return x, (beta, gamma, beta2, gamma2, count_now), states

  def get_initial_state(self, bsize, dim):
    rnn_init = self.rnn_memory.get_initial_state(bsize)
    proto_init = self.proto_memory.get_initial_state(bsize, dim)
    self._num_rnn_states = len(rnn_init)
    self._num_proto_states = len(proto_init)
    h_last = tf.zeros([bsize, dim])
    y_last = tf.zeros([bsize], dtype=tf.int64)
    pred_last = tf.zeros([bsize, self.proto_memory.max_classes + 1])
    return (*rnn_init, *proto_init, y_last, h_last, pred_last)

  @property
  def readout(self):
    return self._readout

  @property
  def rnn_memory(self):
    return self._rnn_memory

  @property
  def proto_memory(self):
    return self._proto_memory
