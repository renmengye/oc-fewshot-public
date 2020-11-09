"""MANN module.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.layer_norm import LayerNorm
from fewshot.models.modules.lstm import LSTM, StackLSTM
from fewshot.models.modules.nnlib import Linear
from fewshot.models.registry import RegisterModule
from fewshot.utils.logger import get as get_logger
from fewshot.models.variable_context import variable_scope

log = get_logger()


@RegisterModule('mann')
class MANN(ContainerModule):

  def __init__(self,
               name,
               in_dim,
               memory_dim,
               controller_dim,
               nslot,
               nread,
               memory_decay,
               controller_type='lstm',
               memory_layernorm=False,
               controller_layernorm=False,
               controller_nstack=2,
               dtype=tf.float32):
    """Initialize a MANN module.

    Args:
      name: String. Name of the module.
      in_dim: Int. Input dimension.
      memory_dim: Int. Memory dimension.
      controller_dim: Int. Hidden dimension for the controller.
      nslot: Int. Number of memory slots.
      nread: Int. Number of read heads.
      memory_decay: Float. Memory decay coefficient.
      controller_type: String. `lstm` or `stack_lstm`.
      dtype: Data type.
    """
    super(MANN, self).__init__(dtype=dtype)
    self._in_dim = in_dim
    self._memory_dim = memory_dim
    self._controller_dim = controller_dim
    self._nslot = nslot
    self._nread = nread
    self._controller_nstack = controller_nstack
    self._controller_type = controller_type
    with variable_scope(name):
      if controller_layernorm:
        log.info('Using LayerNorm in controller module.')
      if controller_type == 'lstm':
        self._controller = LSTM(
            "controller_lstm",
            in_dim,
            controller_dim,
            layernorm=controller_layernorm,
            dtype=dtype)
      elif controller_type == 'stack_lstm':
        log.info('Use {}-stack LSTM'.format(controller_nstack))
        self._controller = StackLSTM(
            "stack_controller_lstm",
            in_dim,
            controller_dim,
            controller_nstack,
            layernorm=controller_layernorm,
            dtype=dtype)
      rnd = np.random.RandomState(0)
      self._rnd = rnd
      self._gamma = memory_decay
      D = memory_dim
      N = nread
      M = nslot
      self._memory_init = 1e-5 * tf.ones(
          [M, D], name="memory_init", dtype=dtype)

      def ctrl2mem_bias_init():
        zeros = tf.zeros([2 * N * D], dtype=self.dtype)
        ones = -2.0 * tf.ones([N], dtype=self.dtype)
        return tf.concat([zeros, ones], axis=0)

      self._ctrl2mem = Linear(
          "ctrl2mem",
          controller_dim,
          2 * nread * memory_dim + nread,
          b_init=ctrl2mem_bias_init)
      self._temp = tf.Variable(1.0, name="temp", dtype=dtype)
      if memory_layernorm:
        log.info('Using LayerNorm for each memory iteration.')
        self._mem_layernorm = LayerNorm("memory_layernorm", D, dtype=dtype)
      else:
        self._mem_layernorm = None

  def forward(self, x, ctrl_c, ctrl_h, memory, usage, read_vec_last, t):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      h_last: Dictionary, see `get_initial_state`.
      write: Bool. Whether enable write.
    """
    B = tf.constant(x.shape[0])
    N = self.nread
    M = self.nslot
    D = self.memory_dim
    ctrl_out, (ctrl_c, ctrl_h) = self._controller(x, ctrl_c, ctrl_h)
    ctrl_mem = self._ctrl2mem(ctrl_out)  # [B, 3 * N * D]
    key = tf.reshape(ctrl_mem[:, :N * D], [-1, N, D])  # [B, N * D]
    write_content = tf.reshape(ctrl_mem[:, N * D:2 * N * D],
                               [-1, N, D])  # [B, N * D]
    sigma = tf.reshape(ctrl_mem[:, 2 * N * D:], [-1, N])  # [B, N]
    read_vec = self.similarity(key, memory)  # [B, N, M]
    read_vec *= self._temp  # Temperature term.
    read_vec = tf.nn.softmax(read_vec)  # [B, N, M]

    # Read memory content
    y = tf.reduce_sum(
        tf.expand_dims(read_vec, -1) * tf.expand_dims(memory, 1),
        [1, 2])  # [B, D]

    # Write memory content
    gate = tf.expand_dims(tf.math.sigmoid(sigma), -1)  # [B, N, 1]
    _, usage_idx = tf.nn.top_k(-usage, M)  # [B, N]
    usage_idx = usage_idx[:, :N]
    least_used = tf.one_hot(usage_idx, M)  # [B, N, M]
    write_vec = gate * read_vec_last + (1 - gate) * least_used  # [B, N, M]

    # Clear usage at LU location.
    all_least_used = tf.reduce_sum(least_used, [1])  # [B, M]
    all_read = tf.reduce_sum(read_vec, [1])  # [B, M]
    all_write = tf.reduce_sum(write_vec, [1])
    usage *= 1.0 - tf.reduce_sum(least_used, [1])
    usage = usage * self._gamma + all_read + all_write  # [B, M]

    # Clear memory at LU location.
    memory *= tf.expand_dims((1.0 - all_least_used), -1)  # [B, M, 1]
    # Write new content.
    memory += tf.reduce_sum(
        tf.expand_dims(write_vec, -1) * tf.expand_dims(write_content, 2),
        [1])  # [B, M, D]

    # Optional memory layer norm.
    if self._mem_layernorm is not None:
      memory = self._mem_layernorm(memory)

    # Time step
    t += tf.constant(1)
    return y, (ctrl_c, ctrl_h, memory, usage, read_vec, t)

  def get_initial_state(self, bsize):
    """Initialize hidden state."""
    ctrl_c, ctrl_h = self._controller.get_initial_state(bsize)
    memory = tf.tile(tf.expand_dims(self._memory_init, 0), [bsize, 1, 1])
    usage = tf.zeros([bsize, self.nslot], dtype=self.dtype)
    read_vec = tf.zeros([bsize, self.nread, self.nslot], dtype=self.dtype)
    t = tf.constant(0, dtype=tf.int32)
    return ctrl_c, ctrl_h, memory, usage, read_vec, t

  def _expand(self, num, x):
    """Expand one variable."""
    tile = [1] + [num] + [1] * (len(x.shape) - 1)
    reshape = [-1] + list(x.shape[1:])
    return tf.reshape(tf.tile(tf.expand_dims(x, 1), tile), reshape)

  def expand_state(self, num, ctrl_c, ctrl_h, memory, usage, read_vec, t):
    """Expand the hidden state for query set."""
    ctrl_c = self._expand(ctrl_c)
    ctrl_h = self._expand(ctrl_h)
    memory = self._expand(memory)
    usage = self._expand(usage)
    read_vec = self._expand(read_vec)
    return ctrl_c, ctrl_h, memory, usage, read_vec, t

  def similarity(self, key, memory):
    """Query the memory with a key using cosine similarity.

    Args:
      key: [B, N, D]. B: batch size, N: number of reads, D: dimension.
      memory: [B, M, D]. B: batch size, M: number of slots, D: dimension.

    Returns:
      sim: [B, N, M]
    """
    eps = 1e-7
    key_norm = tf.sqrt(tf.reduce_sum(tf.square(key), [-1],
                                     keepdims=True))  # [B, N, 1]
    key_ = key / (key_norm + eps)  # [B, N, D]
    memory_norm = tf.sqrt(
        tf.reduce_sum(tf.square(memory), [-1], keepdims=True))  # [B, M, 1]
    memory_ = memory / (memory_norm + eps)  # [B, M, D]
    sim = tf.reduce_sum(
        tf.expand_dims(key_, 2) * tf.expand_dims(memory_, 1),
        [-1])  # [B, N, M]
    return sim

  def end_iteration(self, h_last):
    """End recurrent iterations."""
    return h_last

  @property
  def in_dim(self):
    return self._in_dim

  @property
  def memory_dim(self):
    return self._memory_dim

  @property
  def nin(self):
    return self._in_dim

  @property
  def nout(self):
    return self._memory_dim

  @property
  def controller_dim(self):
    return self._controller_dim

  @property
  def nslot(self):
    return self._nslot

  @property
  def nread(self):
    return self._nread
