"""Unit tests for semisupervised sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import unittest

from fewshot.data.samplers.crp_sampler import CRPSampler
from fewshot.data.samplers.semisupervised_episode_sampler import SemiSupervisedEpisodeSampler  # NOQA


class Dataset():

  def __init__(self, n, m):
    self._cls_dict = self._get_cls_dict(n, m)

  def get_cls_dict(self):
    return self._cls_dict

  def _get_cls_dict(self, n, m):
    """Gets a class dict with n classes and m images per class."""
    cls_dict = {}
    counter = 0
    for k in range(n):
      cls_dict[k] = np.arange(counter, counter + m)
      counter += m
    return cls_dict


class SemiSupervisedEpisodeSamplerTests(unittest.TestCase):

  def test_basic(self):
    n = 20
    m = 10
    sampler = CRPSampler(0)
    sampler = SemiSupervisedEpisodeSampler(sampler, 0)
    sampler.set_dataset(Dataset(n, m))
    for x in range(100):
      collection = sampler.sample_collection(
          10,
          2,
          alpha=0.3,
          theta=1.0,
          max_num_per_cls=m,
          max_num=40,
          label_ratio=0.5)
      s, q = collection['support'], collection['query']
      print('Support', s)
      print('Query', q)

  def test_distractor_basic(self):
    n = 20
    m = 10
    sampler = CRPSampler(0)
    sampler = SemiSupervisedEpisodeSampler(sampler, 0)
    sampler.set_dataset(Dataset(n, m))
    print('start')
    print(sampler.cls_dict)
    for x in range(100):
      print(x)
      collection = sampler.sample_collection(
          10,
          2,
          nd=5,
          sd=3,
          md=2,
          alpha=0.3,
          theta=1.0,
          max_num_per_cls=m,
          max_num=40,
          label_ratio=0.5)
      s, q = collection['support'], collection['query']


if __name__ == '__main__':
  unittest.main()
