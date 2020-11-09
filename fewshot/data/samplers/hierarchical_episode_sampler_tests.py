"""Unit tests for hierarchical sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import unittest

from fewshot.data.samplers.crp_sampler import CRPSampler
from fewshot.data.samplers.hierarchical_episode_sampler import HierarchicalEpisodeSampler  # NOQA
from fewshot.data.samplers.blender import BlurBlender
from fewshot.data.samplers.blender import MarkovSwitchBlender


class Dataset():

  def __init__(self, k, n, m):
    self._cls_dict = self._get_cls_dict(n, m)
    self._hierarchy_dict = self._get_hierarchy_dict(k, n)

  def get_cls_dict(self):
    return self._cls_dict

  def _get_cls_dict(self, n, m):
    """Gets a class dict with n classes and m images per class."""
    cls_dict = {}
    counter = 0
    for n_ in range(n):
      cls_dict[n_] = np.arange(counter, counter + m)
      counter += m
    return cls_dict

  def get_hierarchy_dict(self):
    return self._hierarchy_dict

  def _get_hierarchy_dict(self, k, n):
    """Gets a hierarchy dict with k families and n classes."""
    nperk = n // k
    hdict = {}
    counter = 0
    for k_ in range(k):
      hdict[k_] = np.arange(counter, counter + nperk)
      counter += nperk
    return hdict


class HierarchicalEpisodeSamplerTests(unittest.TestCase):

  def test_basic(self):
    k = 10
    n = 100
    m = 100
    subsampler = CRPSampler(0)
    blender = BlurBlender(window_size=20, stride=5, nrun=10, seed=0)
    sampler = HierarchicalEpisodeSampler(subsampler, blender, False, 0)
    sampler.set_dataset(Dataset(k, n, m))
    for _ in range(10):
      collection = sampler.sample_collection(
          50,
          2,
          alpha=0.5,
          theta=1.0,
          nstage=5,
          max_num=100,
          max_num_per_cls=m)
    print(collection['support'])

  def test_markov(self):
    k = 10
    n = 100
    m = 100
    subsampler = CRPSampler(0)
    blender = MarkovSwitchBlender(np.ones([3]) / 3.0, 0.5, 0)
    sampler = HierarchicalEpisodeSampler(subsampler, blender, False, 0)
    sampler.set_dataset(Dataset(k, n, m))
    collection = sampler.sample_collection(
        30, 2, alpha=0.5, theta=1.0, nstage=3, max_num=60, max_num_per_cls=m)
    print(collection['support'])

  def test_markov_hierarchy(self):
    k = 10
    n = 100
    m = 100
    subsampler = CRPSampler(0)
    blender = MarkovSwitchBlender(np.ones([3]) / 3.0, 0.5, 0)
    sampler = HierarchicalEpisodeSampler(subsampler, blender, True, 0)
    sampler.set_dataset(Dataset(k, n, m))
    collection = sampler.sample_collection(
        30, 2, alpha=0.5, theta=1.0, nstage=3, max_num=60, max_num_per_cls=m)
    print(collection['support'])


if __name__ == '__main__':
  unittest.main()
