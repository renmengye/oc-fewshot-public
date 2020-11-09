"""Unit tests for data augmentation preprocessor."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import unittest

from fewshot.data.preprocessors import DataAugmentationPreprocessor


class DataAugmentationTests(unittest.TestCase):

  def test_basic(self):
    p = DataAugmentationPreprocessor(32, 36, True, True, False)
    x = np.random.uniform(0.0, 1.0, [16, 32, 32, 3])
    y = p.preprocess(x)
    print(y)


if __name__ == '__main__':
  unittest.main()
