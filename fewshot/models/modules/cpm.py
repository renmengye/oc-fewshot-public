"""Contextual Prototypical Memory: ProtoNet memory plus an RNN module.
Note that this version does not contain semisupervised write.
For semisupervised version, see cpm_ssl.py

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


@RegisterModule('cpm')
@RegisterModule('proto_plus_rnn_v4')
class CPM(ContainerModule):

  def __init__(self,
               name,
               proto_memory,
               rnn_memory,
               readout_type='linear',
               use_pred_beta_gamma=True,
               use_feature_fuse=True,
               use_feature_fuse_gate=True,
               use_feature_scaling=True,
               use_feature_memory_only=False,
               skip_unk_memory_update=False,
               dtype=tf.float32):
    super(CPM, self).__init__(dtype=dtype)
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

    D = self._rnn_memory.in_dim
    D_in = self._rnn_memory.memory_dim
    self._dim = D

    # h        [D]
    # scale    [D]
    # gamma    [1]
    # beta     [1]
    # x_gate   [1]
    # h_gate   [1]
    bias_init = [
        tf.zeros([D]),
        tf.zeros([D]),
        tf.zeros([1]),
        tf.zeros([1]) + proto_memory._radius_init,
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
    proto_states = states[self._num_rnn_states:-2]
    y_last = states[-2]
    h_last = states[-1]
    # Input is the same as the input image. RNN doesn't see the label. ----
    x_rnn = x
    # ---------------------------------------------------------------------
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

    if store:
      proto_states = self._proto_memory.store(h_last, y_last, *proto_states)
    readout = self.readout(rnn_out)
    h_gate = tf.math.sigmoid(readout[:, -1:])
    x_gate = tf.math.sigmoid(readout[:, -2:-1])
    beta = readout[:, -3]
    gamma = tf.nn.softplus(readout[:, -4] + 1.0)
    h = readout[:, :D]
    scale = tf.math.softplus(readout[:, D:2 * D])
    if self._use_feature_fuse:
      log.info('Using feature fuse')
      if self._use_feature_fuse_gate:
        log.info('Using feature fuse gating')
        x = x * x_gate + h * h_gate
      else:
        log.info('Not using feature fuse gating')
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
    h_last = x
    y_last = tf.cast(y, y_last.dtype)
    states = (*rnn_states, *proto_states, y_last, h_last)
    return y_cls, y_cls[:, -1], rnn_out, states

  def get_initial_state(self, bsize):
    D = self.proto_memory.dim
    rnn_init = self.rnn_memory.get_initial_state(bsize)
    proto_init = self.proto_memory.get_initial_state(bsize)
    self._num_rnn_states = len(rnn_init)
    self._num_proto_states = len(proto_init)
    h_last = tf.zeros([bsize, D])
    y_last = tf.zeros([bsize], dtype=tf.int64)
    return (*rnn_init, *proto_init, y_last, h_last)

  @property
  def readout(self):
    return self._readout

  @property
  def rnn_memory(self):
    return self._rnn_memory

  @property
  def proto_memory(self):
    return self._proto_memory
