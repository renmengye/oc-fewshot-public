"""LSTM modules.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.registry import RegisterModule
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Linear
from fewshot.models.modules.layer_norm import LayerNorm
from fewshot.models.variable_context import variable_scope


@RegisterModule('lstm')
class LSTM(ContainerModule):
  """A standard LSTM module."""

  def __init__(self, name, nin, nout, layernorm=False, dtype=tf.float32):
    super(LSTM, self).__init__(dtype=dtype)
    self._nin = nin
    self._nout = nout
    self._layernorm = layernorm

    def _b_init():
      return tf.concat(
          [tf.ones([nout], dtype=dtype),
           tf.zeros([3 * nout], dtype=dtype)],
          axis=0)

    with variable_scope(name):
      self._gates = Linear(
          "gates_linear", nin + nout, 4 * nout, b_init=_b_init)
      if layernorm:
        self._ln = LayerNorm("layernorm", 4 * nout, dtype=dtype)

  def forward(self, x, c_last, h_last):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      c_last: [B, D]. Cell states of the previous time step.
      h_last: [B, D]. Hidden states of the previous time step.

    Returns:
      A tuple of output and the hidden states.
    """
    x_comb = tf.concat([x, h_last], axis=-1)
    gates = self._gates(x_comb)
    if self._layernorm:
      gates = self._ln(gates)
    D = self.nout
    f_gate = tf.sigmoid(gates[:, :D])
    i_gate = tf.sigmoid(gates[:, D:2 * D])
    o_gate = tf.sigmoid(gates[:, 2 * D:3 * D])
    c_gate = tf.tanh(gates[:, 3 * D:])
    c = c_last * f_gate + i_gate * c_gate
    h = o_gate * tf.tanh(c)
    return h, (c, h)

  def end_iteration(self, h_last):
    """End recurrent iterations."""
    return h_last

  def get_initial_state(self, bsize):
    return (tf.zeros([bsize, self.nout], dtype=self.dtype),
            tf.zeros([bsize, self.nout], dtype=self.dtype))

  @property
  def nin(self):
    return self._nin

  @property
  def nout(self):
    return self._nout

  @property
  def in_dim(self):
    return self._nin

  @property
  def memory_dim(self):
    return self._nout


@RegisterModule('stack_lstm')
class StackLSTM(ContainerModule):
  """An RNN with stacked LSTM cells."""

  def __init__(self,
               name,
               nin,
               nout,
               nstack,
               layernorm=False,
               dtype=tf.float32):
    super(StackLSTM, self).__init__(dtype=dtype)
    self._nin = nin
    self._nout = nout
    self._lstm_list = []
    self._nstack = nstack
    assert nstack > 1, 'Number of LSTM > 1'
    with variable_scope(name):
      for n in range(nstack):
        self._lstm_list.append(
            LSTM(
                "cell_{}".format(n),
                nin,
                nout,
                layernorm=layernorm,
                dtype=dtype))
        nin = nout

  def forward(self, x, c_last, h_last):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      c_last: [N, B, D]. Cell states for last time step.
      h_last: [N, B, D]. Hidden states for last time step.
    """
    c_list = tf.TensorArray(self.dtype, size=self._nstack)
    h_list = tf.TensorArray(self.dtype, size=self._nstack)
    for n in range(self._nstack):
      lstm_ = self._lstm_list[n]
      x, (c_new, h_new) = lstm_(x, c_last[n], h_last[n])
      c_list = c_list.write(n, c_new)
      h_list = h_list.write(n, h_new)
    return x, (c_list.stack(), h_list.stack())

  def get_initial_state(self, bsize):
    """Computes the initial state."""
    c_list = tf.TensorArray(self.dtype, size=self._nstack)
    h_list = tf.TensorArray(self.dtype, size=self._nstack)
    for n in range(self._nstack):
      lstm_ = self._lstm_list[n]
      c_0, h_0 = lstm_.get_initial_state(bsize)
      c_list = c_list.write(n, c_0)
      h_list = h_list.write(n, h_0)
    return c_list.stack(), h_list.stack()

  def end_iteration(self, h_last):
    """End recurrent iterations."""
    return h_last

  @property
  def nin(self):
    return self._nin

  @property
  def nout(self):
    return self._nout

  @property
  def in_dim(self):
    return self._nin

  @property
  def memory_dim(self):
    return self._nout


@RegisterModule('stack_res_lstm')
class StackResLSTM(StackLSTM):
  """An RNN with stacked LSTM cells with residual connections."""

  def __init__(self, name, nin, nout, nstack, dtype=tf.float32):
    super(StackResLSTM, self).__init__(name, nin, nout, nstack, dtype=dtype)

  def forward(self, x, h_last):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      h_last: List of hidden states for each cell.
    """
    h_list = []
    if h_last is None:
      h_last = self.get_initial_state(int(x.shape[0]))

    for i, (lstm_, h_last_) in enumerate(zip(self._lstm_list, h_last)):
      x_, h_last_new_ = lstm_(x, h_last_)
      if i == 0:
        x = x_
      else:
        x += x_
      h_list.append(h_last_new_)
    return x, h_list
