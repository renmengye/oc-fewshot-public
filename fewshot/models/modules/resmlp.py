"""Multi-layer perceptron with residual connection.
Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Linear
from fewshot.models.variable_context import variable_scope


class ResMLP(ContainerModule):

  def __init__(self, name, layer_size, act_func, bias_init, dtype=tf.float32):
    super(ResMLP, self).__init__(dtype=dtype)
    self._layers = []
    self._layer_size = layer_size
    self._act_func = act_func
    with variable_scope(name):
      for i in range(len(layer_size) - 1):

        if bias_init[i] is None:
          bias_init_ = tf.zeros([layer_size[i + 1]], dtype=dtype)
        else:
          bias_init_ = bias_init[i]

        def bi():
          return bias_init_

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
    for i, (l, a) in enumerate(zip(self._layers, self._act_func)):
      x_old = x
      x = l(x)
      if i > 0:
        x += x_old
      if i < len(self._layer_size) - 2:
        x = a(x)
    return x
