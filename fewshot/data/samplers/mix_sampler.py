"""Mixing a collection of samplers.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


class MixSampler(object):

  def __init__(self, sampler_list, dist, seed):
    """Initialize a mixed sampler.

    Args:
      sampler_list: A list of samplers.
      dist: Base distribution of samplers.
    """
    self._sampler_list = sampler_list
    self._dist = dist
    assert len(dist) == len(sampler_list)
    assert len(sampler_list) > 1
    self._rnd = np.random.RandomState(seed)

  def sample_collection(self, *args, **kwargs):
    s = np.argmax(self._rnd.multinomial(1, self._dist, size=1))
    return self._sampler_list[s].sample_collection(*args, **kwargs)

  def set_dataset(self, dataset):
    for s in self._sampler_list:
      s.set_dataset(dataset)

  def reset(self):
    """Reset randomness"""
    for s in self._sampler_list:
      s.reset()
