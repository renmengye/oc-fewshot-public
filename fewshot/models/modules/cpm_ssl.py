"""Contextual Prototypical Memory: ProtoNet memory plus an RNN module.
Note that this is the semisupervised version.
This should be compatible to fully supervised sequence as well.

Use the RNN module to
  1) encode the examples (with context);
  2) output unknown.

Use the ProtoNet to
  1) decode the class;
  2) store class prototypes.

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


@RegisterModule('cpm_ssl')
@RegisterModule('proto_plus_rnn_ssl_v4')
class CPMSSL(ContainerModule):

  def __init__(self, name, proto_memory, rnn_memory, config, dtype=tf.float32):
    super(CPMSSL, self).__init__(dtype=dtype)
    self._rnn_memory = rnn_memory
    self._proto_memory = proto_memory

    # ------------- Feature Fusing Capability Ablation --------------
    self._use_pred_beta_gamma = config.use_pred_beta_gamma  # CHECK
    self._use_feature_fuse = config.use_feature_fuse  # CHECK
    self._use_feature_fuse_gate = config.use_feature_fuse_gate  # CHECK
    self._use_feature_scaling = config.use_feature_scaling  # CHECK
    self._use_feature_memory_only = config.use_feature_memory_only  # CHECK

    # ------------- SSL Capability Ablation --------------
    self._skip_unk_memory_update = config.skip_unk_memory_update  # CHECK
    self._use_ssl = config.use_ssl  # CHECK
    self._use_ssl_beta_gamma_write = config.use_ssl_beta_gamma_write  # CHECK
    self._use_ssl_temp = config.use_ssl_temp  # CHECK

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

    if config.readout_type == 'linear':
      log.info("Using linear readout")
      self._readout = Linear('readout', D_in, D_out, b_init=b_init)
    elif config.readout_type == 'mlp':
      log.info("Using MLP readout")
      self._readout = MLP(
          'readout_mlp', [D_in, D_out, D_out],
          bias_init=[None, bias_init],
          act_func=[tf.math.tanh])
    elif config.readout_type == 'resmlp':
      log.info("Using ResMLP readout")
      # -----AFFINE--------
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

    if self._skip_unk_memory_update:
      log.info('Skip UNK example RNN storage')

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
      log.info('Disabling SSL compacity')
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
      D2 = proto_states[0].shape[-1]
      storage_new.set_shape(
          [x.shape[0], self._proto_memory.max_classes + 1, D2])
      count_new.set_shape([x.shape[0], self._proto_memory.max_classes + 1])
      proto_states = (storage_new, count_new)

    if self._use_feature_fuse:
      log.info('Using feature fuse')
      if self._use_feature_fuse_gate:
        log.info('Using feature fuse gate')
        x = x * x_gate + h * h_gate
      else:
        log.info('Not using feature fuse gate')
        x = x + h
    else:
      log.info('Not using feature fuse')

    if self._use_feature_memory_only:
      log.info('Only using memory feature')
      x = h

    storage, count = proto_states

    if self._use_feature_scaling:
      log.info('Using feature scaling')
      storage_scale = storage * scale[:, None, :]
      x_scale = x * scale
    else:
      log.info('Not using feature scaling')
      storage_scale = storage
      x_scale = x

    if self._use_pred_beta_gamma:
      log.info('Using self predicted beta/gamma')
    else:
      log.info('Not using self predicted beta/gamma')
      beta = None
      gamma = None
    y_cls = self.proto_memory.retrieve(
        x_scale, storage_scale, count, t, beta=beta, gamma=gamma, add_new=True)

    if self._use_pred_beta_gamma:
      if self._use_ssl_beta_gamma_write:
        log.info('Using self predicted beta2/gamma2 for SSL')
      else:
        log.info('Not using self predicted beta2/gamma2 for SSL')
        beta2 = beta
        gamma2 = gamma
    else:
      log.info('Using static beta2/gamma2 for SSL')
      if self._use_ssl_beta_gamma_write:
        beta2 = self.proto_memory._beta2
        gamma2 = self.proto_memory._gamma2
      else:
        beta2 = None
        gamma2 = None

    if self._use_ssl_temp:
      log.info('Using extra temperature for SSL')
    else:
      log.info('Not using extra temperature for SSL')
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
    return y_cls, y_cls[:, -1], rnn_out, states

  def get_initial_state(self, bsize):
    rnn_init = self.rnn_memory.get_initial_state(bsize)
    proto_init = self.proto_memory.get_initial_state(bsize)
    self._num_rnn_states = len(rnn_init)
    self._num_proto_states = len(proto_init)
    h_last = tf.zeros([bsize, self.proto_memory.dim])
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
