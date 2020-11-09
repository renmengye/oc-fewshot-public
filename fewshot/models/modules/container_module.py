"""A module base class that contains other sub-modules.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect
import tensorflow as tf
from collections import OrderedDict

from fewshot.models.modules.module import Module
from fewshot.models.modules.weight_module import WeightModule
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ContainerModule(WeightModule):

  def __init__(self, dtype=tf.float32):
    super(ContainerModule, self).__init__(dtype=dtype)
    self._modules = None
    self._subweights = None

  def get_modules(self):
    """Look for submodules in class members."""
    modules = OrderedDict()
    member_list = sorted(inspect.getmembers(self), key=lambda x: x[0])
    for m in member_list:
      if isinstance(m[1], Module):
        # print(m[0], m[1])
        modules[m[1]] = None
      elif isinstance(m[1], list):  # Enumerate everything in the list.
        # print(m[0], m[1])
        for j, m_ in enumerate(m[1]):
          if isinstance(m_, Module):
            modules[m_] = None
    # print('modules', modules)
    return list(modules.keys())
    # return list(modules)

  def modules(self):
    """Modules that belong to this instance."""
    # Compute the sub-modules for one-time.
    if self._modules is None:
      self._modules = self.get_modules()
    return self._modules

  def weights(self):
    """All parameters that belong to this instance."""
    if self._subweights is None:
      weights = self._weights
      for m in self.modules():
        if isinstance(m, WeightModule):
          weights.extend(m.weights())
      self._subweights = weights
    return self._subweights

  def set_trainable(self, trainable):
    """Set parameter trainable status."""
    for m in self.modules():
      m.set_trainable(trainable)
    for w in self.weights():
      # for w in self.weights():
      w._trainable = trainable
      log.info('Set {} trainable={}'.format(w.name, trainable))
