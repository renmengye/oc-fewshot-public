from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.registry import RegisterModule
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Linear
from fewshot.models.modules.layer_norm import LayerNorm
from fewshot.models.variable_context import variable_scope


@RegisterModule('gru')
class GRU(ContainerModule):
  """Gated recurrent unit"""

  def __init__(self, name, nin, nout, dtype=tf.float32):
    super(GRU, self).__init__(dtype=dtype)
    self._nin = nin
    self._nout = nout

    with variable_scope(name):
      self._gates = Linear("gates_linear", nin + nout, 2 * nout)
      self._linear = Linear("linear", nin + nout, nout)

  def forward(self, x, h_last):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      h_last: [B, D]. Hidden states of the previous timestep.

    Returns:
      h
    """
    D = self.nout
    x_comb = tf.concat([x, h_last], axis=-1)
    gates = self._gates(x_comb)
    r_gate = tf.math.sigmoid(gates[:, :D])
    z_gate = tf.math.sigmoid(gates[:, D:])
    h_hat = tf.math.tanh(self._linear(tf.concat([x, h_last * r_gate])))
    h = (1.0 - z_gate) * h_hat + z_gate * h_hat
    return h

  def end_iteration(self, h_last):
    return h_last

  def get_initial_state(self, bsize):
    return tf.zeros([bsize, self.nout], dtype=self.dtype)

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



@RegisterModule('gau')
class GAU(ContainerModule):
  """GRU with 1-d gates and without activation"""

  def __init__(self,
               name,
               nin,
               nout,
               layernorm=False,
               bias_init=-2.0,
               dtype=tf.float32):
    super(GAU, self).__init__(dtype=dtype)
    self._nin = nin
    self._nout = nout
    self._layernorm = layernorm
    self._gates = Linear(
        "gates_linear", nin + nout, 1, b_init=lambda: tf.ones(1) * bias_init)
    if layernorm:
      self._ln = LayerNorm("layernorm", nin + nout, dtype=dtype)

  def forward(self, x, h_last):
    """Forward one timestep.

    Args:
      x: [B, D]. Input.
      h_last: [B, D]. Hidden states of the previous timestep.

    Returns:
      h
    """
    x_comb = tf.concat([x, h_last], axis=-1)
    if self._layernorm:
      x_comb = self._ln(x_comb)
    gates = self._gates(x_comb)
    f_gate = tf.math.sigmoid(gates)
    # tf.print('f gate', f_gate)
    h = (1.0 - f_gate) * h_last + f_gate * x
    return h, h

  def end_iteration(self, h_last):
    return h_last

  def get_initial_state(self, bsize):
    return tf.zeros([bsize, self.nout], dtype=self.dtype)

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

@RegisterModule('lstm1dmod')
class LSTM1DMod(ContainerModule):
  """A standard LSTM module."""

  def __init__(self, name, nin, nout, dtype=tf.float32):
    super(LSTM1DMod, self).__init__(dtype=dtype)
    self._nin = nin
    self._nout = nout

    with variable_scope(name):
      self._gates = Linear("gates_linear", nin + nout, nout + 2)
      # self._gates2 = Linear("gates_linear", nout, nout)

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
    D = self.nout
    f_gate = tf.sigmoid(gates[:, :1])
    i_gate = tf.sigmoid(gates[:, 1:2])
    # o_gate = tf.sigmoid(gates[:, 2:2 + D])
    o_gate = tf.sigmoid(gates[:, 2:3])
    # c = c_last * f_gate +  x * i_gate
    c = c_last * f_gate + x * (1 - f_gate)
    h = o_gate * tf.tanh(c)
    # h = tf.tanh(c2)
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
