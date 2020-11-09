"""Group normalization unit tests."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import unittest

from fewshot.models.modules.group_norm import GroupNorm


class GroupNormTests(unittest.TestCase):

  def test_basic(self):
    np.random.seed(0)
    x = np.random.uniform(-1.0, 1.0, [10, 4, 4, 16]).astype(np.float32)
    x = tf.constant(x)
    gn = GroupNorm('gn', 16, 8, 'NHWC')
    gn(x)
    x = np.random.uniform(-1.0, 1.0, [10, 16, 4, 4]).astype(np.float32)
    x = tf.constant(x)
    gn = GroupNorm('gn', 16, 8, 'NCHW')
    gn(x)


if __name__ == '__main__':
  unittest.main()
