"""Unit tests for sequential CRP sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import unittest
from fewshot.data.samplers.seq_crp_sampler import SeqCRPSampler


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


class SeqCRPSamplerTests(unittest.TestCase):

  def test_basic(self):
    n = 20
    m = 10
    sampler = SeqCRPSampler(0)
    sampler.set_dataset(Dataset(n, m))
    for x in range(100):
      collection = sampler.sample_collection(
          10, 2, stages=2, alpha=0.3, theta=1.0, max_num_per_cls=m)


if __name__ == '__main__':
  unittest.main()
