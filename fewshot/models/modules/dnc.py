"""DNC module without link.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.lstm import LSTM, StackLSTM
from fewshot.models.modules.nnlib import Linear
from fewshot.models.modules.mlp import MLP
from fewshot.models.modules.layer_norm import LayerNorm
from fewshot.models.registry import RegisterModule
from fewshot.utils.logger import get as get_logger
from fewshot.models.variable_context import variable_scope

log = get_logger()


@RegisterModule('dnc')
@RegisterModule('mann_writeattn_nwrite_decay')
class DNC(ContainerModule):

  def __init__(self, name, in_dim, config, dtype=tf.float32):
    """Initialize a DNC module.

    Args:
      name: String. Name of the module.
      in_dim: Int. Input dimension.
      memory_dim: Int. Memory dimension.
      controller_dim: Int. Hidden dimension for the controller.
      nslot: Int. Number of memory slots.
      nread: Int. Number of read heads.
      nwrite: Int. Number of write heads.
      controller_type: String. `lstm` or `mlp.
      memory_layernorm: Bool. Whether perform LayerNorm on each memory
        iteration.
      dtype: Data type.
    """
    super(DNC, self).__init__(dtype=dtype)
    log.info('Currently using MANN with separate write attention')
    log.info('Currently using MANN with decay')
    self._in_dim = in_dim
    self._memory_dim = config.memory_dim
    self._controller_dim = config.controller_dim
    self._nslot = config.num_slots
    self._nread = config.num_reads
    self._nwrite = config.num_writes
    self._controller_nstack = config.controller_nstack
    self._controller_type = config.controller_type
    self._similarity_type = config.similarity_type
    with variable_scope(name):
      if config.controller_layernorm:
        log.info('Using LayerNorm in controller module.')
      if config.controller_type == 'lstm':
        self._controller = LSTM(
            "controller_lstm",
            in_dim,
            config.controller_dim,
            layernorm=config.controller_layernorm,
            dtype=dtype)
      elif config.controller_type == 'stack_lstm':
        log.info('Use {}-stack LSTM'.format(config.controller_nstack))
        self._controller = StackLSTM(
            "stack_controller_lstm",
            in_dim,
            config.controller_dim,
            config.controller_nstack,
            layernorm=config.controller_layernorm,
            dtype=dtype)
      elif config.controller_type == 'mlp':
        log.info('Use MLP')
        self._controller = MLP(
            "controller_mlp",
            [in_dim, config.controller_dim, config.controller_dim],
            layernorm=config.controller_layernorm,
            dtype=dtype)
      rnd = np.random.RandomState(0)
      self._rnd = rnd
      self._memory_init = 1e-5 * tf.ones([config.num_slots, config.memory_dim],
                                         name="memory_init",
                                         dtype=dtype)

      # N. Item name         Shape    Init    Comment
      # ------------------------------------------------------------
      # 1) read query        N x D    0.0
      # 2) write query       Nw x D   0.0
      # 3) write content     Nw x D   0.0
      # 4) forget gate       N        -2.0    No forget after read
      # 5) write gate        Nw       2.0     Always write
      # 6) interp gate       Nw       -2.0    Always use LRU
      # 7) read temp         N        0.0     Default 1.0
      # 8) write temp        Nw       0.0     Default 1.0
      # 9) erase             M        -2.0    Default no erase
      Nr = self._nread
      Nw = self._nwrite
      D = self._memory_dim
      M = self._nslot

      def ctrl2mem_bias_init():
        AA = tf.zeros([Nr * D + 2 * Nw * D], dtype=self.dtype)
        BB = -2.0 * tf.ones([Nr], dtype=self.dtype)
        CC = 2.0 * tf.ones([Nw], dtype=self.dtype)
        DD = -2.0 * tf.ones([Nw], dtype=self.dtype)
        EE = 0.0 * tf.ones([Nr], dtype=self.dtype)
        FF = 0.0 * tf.ones([Nw], dtype=self.dtype)
        GG = -2.0 * tf.ones([M], dtype=self.dtype)
        return tf.concat([AA, BB, CC, DD, EE, FF, GG], axis=0)

      self._ctrl2mem = Linear(
          "ctrl2mem",
          config.controller_dim,
          Nr * D + 2 * Nw * D + Nr + 2 * Nw + Nr + Nw + M,
          b_init=ctrl2mem_bias_init)
      if config.memory_layernorm:
        log.info('Using LayerNorm for each memory iteration.')
        self._mem_layernorm = LayerNorm("memory_layernorm", D, dtype=dtype)
      else:
        self._mem_layernorm = None

  def slice_vec_2d(self, vec, size):
    return tf.split(vec, size, axis=1)

  def oneplus(self, x):
    return 1 + tf.math.log(1 + tf.math.exp(x))

  def get_lru(self, usage, Nw):
    """Get least used soft flag."""
    B = tf.constant(usage.shape[0])
    M = usage.shape[1]
    usage_sort, usage_idx = tf.nn.top_k(-usage, M)  # [B, M] [B, M]
    usage_sort = -usage_sort
    least_used = tf.TensorArray(self.dtype, size=Nw)
    for s in range(Nw):
      AA = tf.zeros([B, s], dtype=self.dtype)
      CC = tf.concat([tf.ones([B, 1], dtype=self.dtype), usage_sort[:, s:]],
                     axis=1)
      BB = tf.math.cumprod(CC, axis=1)[:, :-1]
      least_used_ = tf.concat([AA, BB], axis=-1)  # [B, M]
      least_used_ *= 1.0 - usage_sort
      least_used = least_used.write(s, least_used_)
    least_used = tf.transpose(least_used.stack(), [1, 0, 2])  # [B, Nw, M]

    batch_idx = tf.tile(tf.reshape(tf.range(B), [-1, 1, 1]), [1, Nw, M])
    nw_idx = tf.tile(tf.reshape(tf.range(Nw), [1, -1, 1]), [B, 1, M])
    usage_idx_ = tf.tile(tf.reshape(usage_idx, [-1, 1, M]), [1, Nw, 1])
    unsrt_idx = tf.stack([batch_idx, nw_idx, usage_idx_], -1)  # [B, Nw, M, 3]
    least_used = tf.scatter_nd(unsrt_idx, least_used, [B, Nw, M])
    return least_used

  def content_attention(self, query, key, temp):
    """Attention to memory content.

    Args:
      query: [B, N, D] Query vector.
      key: [B, M, D] Key vector.
      temp: [B, N] Temperature.

    Returns:
      attn: [B, N, M] Attention.
    """
    return tf.nn.softmax(self.similarity(query, key) * temp)

  def forward(self, x, memory, usage, t, *args):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      ctrl_c: [B, D]. Last step controller state.
      ctrl_h: [B, D]. Last step controller state.
      memory: [B, M, D]. Memory slots.
      usage: [B, M]. Memory usage.
      t: Int. Timestep counter.
    """
    B = tf.constant(x.shape[0])
    N = self.nread
    Nw = self.nwrite
    M = self.nslot
    D = self.memory_dim
    if self._controller_type in ['lstm', 'stack_lstm']:
      ctrl_c, ctrl_h = args
      ctrl_out = self._controller(x, ctrl_c, ctrl_h)
      ctrl_out, (ctrl_c, ctrl_h) = ctrl_out
    else:
      ctrl_out = self._controller(x)
    ctrl_mem = self._ctrl2mem(ctrl_out)  # [B, N * D + 2 * Nw * D]
    (read_query, write_query, write_content, forget, write_gate,
     sigma, temp_read, temp_write, erase) = self.slice_vec_2d(
         ctrl_mem, [N * D, Nw * D, Nw * D, N, Nw, Nw, N, Nw, M])
    temp_read = tf.expand_dims(self.oneplus(temp_read), -1)  # [B, N, 1]
    temp_write = tf.expand_dims(self.oneplus(temp_write), -1)  # [B, Nw, 1]
    read_query = tf.reshape(read_query, [-1, N, D])  # [B, N * D]
    write_query = tf.reshape(write_query, [-1, Nw, D])  # [B, Nw * D]
    write_content = tf.reshape(write_content, [-1, Nw, D])  # [B, Nw * D]
    # [B, N, M]
    read_vec = self.content_attention(read_query, memory, temp_read)
    write_vec = self.content_attention(write_query, memory,
                                       temp_write)  # [B, Nw, M]

    forget = tf.expand_dims(tf.math.sigmoid(forget), -1)  # [B, N, 1]
    free = tf.reduce_prod(1.0 - forget * read_vec, [1])  # [B, M]

    # Read memory content
    y = tf.reduce_sum(tf.matmul(read_vec, memory),
                      [1])  # [B, N, M] x [B, M, D] = [B, N, D] => [B, D]

    # Write memory content
    interp_gate = tf.expand_dims(tf.math.sigmoid(sigma), -1)  # [B, Nw, 1]
    write_gate = tf.expand_dims(tf.math.sigmoid(write_gate), -1)  # [B, Nw, 1]
    erase = tf.expand_dims(tf.math.sigmoid(erase), -1)  # [B, M, 1]
    least_used = self.get_lru(usage, Nw)  # [B, Nw, M]

    write_vec = write_gate * (interp_gate * write_vec +
                              (1 - interp_gate) * least_used)  # [B, Nw, M]
    all_write = tf.reduce_max(write_vec, [1])
    usage = (usage + all_write - all_write * usage) * free  # [B, M]

    # Erase memory content. # [B, M, D]
    memory *= (tf.ones([B, M, 1]) - erase * tf.expand_dims(all_write, -1))

    # Write new content.
    memory += tf.matmul(
        write_vec, write_content,
        transpose_a=True)  # [B, Nw, M] x [B, Nw, D] = [B, M, D]

    # Optional memory layer norm.
    if self._mem_layernorm is not None:
      memory = self._mem_layernorm(memory)

    # Time step.
    t += tf.constant(1)

    if self._controller_type in ['lstm', 'stack_lstm']:
      return y, (memory, usage, t, ctrl_c, ctrl_h)
    else:
      return y, (memory, usage, t)

  def get_initial_state(self, bsize):
    """Initialize hidden state."""
    memory = tf.tile(tf.expand_dims(self._memory_init, 0), [bsize, 1, 1])
    usage = tf.zeros([bsize, self.nslot], dtype=self.dtype)
    t = tf.constant(0, dtype=tf.int32)
    if self._controller_type in ['lstm', 'stack_lstm']:
      ctrl_c, ctrl_h = self._controller.get_initial_state(bsize)
      return memory, usage, t, ctrl_c, ctrl_h
    elif self._controller_type in ['mlp']:
      return memory, usage, t
    else:
      assert False

  def _expand(self, num, x):
    """Expand one variable."""
    tile = [1] + [num] + [1] * (len(x.shape) - 1)
    reshape = [-1] + list(x.shape[1:])
    return tf.reshape(tf.tile(tf.expand_dims(x, 1), tile), reshape)

  def expand_state(self, num, ctrl_c, ctrl_h, memory, usage, t):
    """Expand the hidden state for query set."""

    memory = self._expand(num, memory)
    usage = self._expand(num, usage)
    if self._controller_type in ['lstm', 'stack_lstm']:
      ctrl_c = self._expand(num, ctrl_c)
      ctrl_h = self._expand(num, ctrl_h)
      return memory, usage, t, ctrl_c, ctrl_h
    else:
      return memory, usage, t

  def similarity(self, query, key):
    """Query the memory with a key using cosine similarity.

    Args:
      query: [B, N, D]. B: batch size, N: number of reads, D: dimension.
      key: [B, M, D]. B: batch size, M: number of slots, D: dimension.

    Returns:
      sim: [B, N, M]
    """
    eps = 1e-7
    if self._similarity_type == 'cosine':
      q_norm = tf.sqrt(tf.reduce_sum(tf.square(query), [-1],
                                     keepdims=True))  # [B, N, 1]
      q_ = query / (q_norm + eps)  # [B, N, D]
      k_norm = tf.sqrt(tf.reduce_sum(tf.square(key), [-1],
                                     keepdims=True))  # [B, M, 1]
      k_ = key / (k_norm + eps)  # [B, M, D]
      sim = tf.matmul(q_, k_, transpose_b=True)
    elif self._similarity_type == 'dot_product':
      sim = tf.matmul(query, key, transpose_b=True)
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

  @property
  def nwrite(self):
    return self._nwrite
