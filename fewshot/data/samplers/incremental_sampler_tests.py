"""Unit tests for incremental sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import unittest

from fewshot.data.samplers.incremental_sampler import IncrementalSampler


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


class IncrementalSamplerTests(unittest.TestCase):

  def test_nonoverlap(self):
    """Tests that the support set and query set does not overlap."""
    for n in range(1, 10, 3):
      for m in range(10, 20, 3):
        for l in range(4, 5):
          for u in range(5, 6):
            for k in range(2, 3):
              sampler = IncrementalSampler(0)
              sampler.set_dataset(Dataset(n, m))
              collection = sampler.sample_collection(
                  n, k, nshot_min=l, nshot_max=u)
              s, q = collection['support'], collection['query']
              s = set(s)
              q = set(q)
              self.assertEqual(len(s.difference(q)), len(s))


if __name__ == '__main__':
  unittest.main()
