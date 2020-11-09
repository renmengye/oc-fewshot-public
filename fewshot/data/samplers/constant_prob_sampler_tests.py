"""Unit tests for constant prob sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import unittest

from fewshot.data.samplers.constant_prob_sampler import ConstantProbSampler


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


class ConstantProbSamplerTests(unittest.TestCase):

  def test_basic(self):
    n = 10
    m = 10
    sampler = ConstantProbSampler(0)
    sampler.set_dataset(Dataset(n, m))
    s, q = sampler.sample_collection(n, 2, p=0.3)
    print(s)
    print(q)


if __name__ == '__main__':
  unittest.main()
