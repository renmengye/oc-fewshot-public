"""DNC module with write head to feed with the label information.
Now previous image content is also fed into the network write head.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.dnc import DNC
from fewshot.models.modules.nnlib import Linear
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope


def swish(x):
  return tf.math.sigmoid(x) * x


class ResMLP(ContainerModule):

  def __init__(self, name, layer_size, dtype=tf.float32):
    super(ResMLP, self).__init__(dtype=dtype)
    self._layer_size = layer_size
    self._layers = []
    with variable_scope(name):
      for i in range(len(layer_size) - 1):

        def bi():
          return tf.zeros([layer_size[i + 1]], dtype=dtype)

        self._layers.append(
            Linear(
                "layer_{}".format(i),
                layer_size[i],
                layer_size[i + 1],
                b_init=bi,
                add_bias=True,
                dtype=dtype))

  def forward(self, x):
    """Forward pass."""
    for i, layer in enumerate(self._layers):
      x_old = x
      x = layer(x)
      if i < len(self._layer_size) - 2:
        x = swish(x)
      if i > 0:
        x += x_old
    return x


@RegisterModule('dnc_writehead_v2')
@RegisterModule('dnc_writeheadfeed2')
class DNCWriteHeadFeed2(DNC):

  def __init__(self, name, in_dim, label_dim, config, dtype=tf.float32):

    super(DNCWriteHeadFeed2, self).__init__(name, in_dim, config, dtype=dtype)

    self._label_dim = label_dim
    use_mlp = False
    with variable_scope(name):
      if use_mlp:
        self._write_query_mlp = ResMLP(
            'write_query_mlp', [
                in_dim, self._nwrite * self._memory_dim,
                self._nwrite * self._memory_dim,
                self._nwrite * self._memory_dim
            ],
            dtype=dtype)
        self._write_content_mlp = ResMLP(
            'write_content_mlp', [
                in_dim, self._nwrite * self._memory_dim,
                self._nwrite * self._memory_dim,
                self._nwrite * self._memory_dim
            ],
            dtype=dtype)
      else:
        self._write_query_mlp = Linear('write_query_mlp', in_dim,
                                       self._nwrite * self._memory_dim)
        self._write_content_mlp = Linear('write_content_mlp', in_dim,
                                         self._nwrite * self._memory_dim)

  def forward(self, x, memory, usage, t, x_last, *args):
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
    read_vec = self.content_attention(read_query, memory,
                                      temp_read)  # [B, N, M]

    # ------------------------------------------------------------------------
    # Change from here
    label = x[:, -self._label_dim:]  # [B, L]
    in_label = tf.concat([x_last, label], axis=-1)
    write_content2 = tf.reshape(self._write_content_mlp(in_label), [-1, Nw, D])
    write_query2 = tf.reshape(self._write_query_mlp(in_label), [-1, Nw, D])
    write_query += write_query2
    write_content += write_content2
    # Change ends here
    # ------------------------------------------------------------------------
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
    x_last = x[:, :-self._label_dim]

    if self._controller_type in ['lstm', 'stack_lstm']:
      return y, (memory, usage, t, x_last, ctrl_c, ctrl_h)
    else:
      return y, (memory, usage, t, x_last)

  def get_initial_state(self, bsize):
    """Initialize hidden state."""
    memory = tf.tile(tf.expand_dims(self._memory_init, 0), [bsize, 1, 1])
    usage = tf.zeros([bsize, self.nslot], dtype=self.dtype)
    t = tf.constant(0, dtype=tf.int32)
    x_last = tf.zeros([bsize, self._in_dim - self._label_dim],
                      dtype=self.dtype)
    if self._controller_type in ['lstm', 'stack_lstm']:
      ctrl_c, ctrl_h = self._controller.get_initial_state(bsize)
      return memory, usage, t, x_last, ctrl_c, ctrl_h
    elif self._controller_type in ['mlp']:
      return memory, usage, t, x_last
    else:
      assert False
