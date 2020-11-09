"""Basic 4-layer convolution network backbone plus a FC layer.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import tensorflow as tf

from fewshot.models.modules.c4_backbone import C4Backbone
from fewshot.models.modules.backbone import Backbone
from fewshot.models.registry import RegisterModule
# from fewshot.models.variable_context import variable_scope
from fewshot.models.modules.nnlib import Linear
from fewshot.models.modules.mlp import MLP


@RegisterModule("c4_plus_fc_backbone")
class C4PlusFCBackbone(Backbone):

  def __init__(self, config, wdict=None):
    super(C4PlusFCBackbone, self).__init__(config)
    self.backbone = C4Backbone(config)

    if len(self.config.num_fc_dim) > 1:
      self.fc = MLP(
          'fc', [config.num_filters[-1]] + list(self.config.num_fc_dim),
          wdict=wdict)
    else:
      self.fc = Linear(
          'fc', config.num_filters[-1], self.config.num_fc_dim,
          wdict=wdict)  # Hard coded for now.

  def forward(self, x, is_training, **kwargs):
    return self.fc(self.backbone(x, is_training=is_training, **kwargs))
