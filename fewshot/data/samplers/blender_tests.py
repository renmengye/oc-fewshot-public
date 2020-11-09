""""Unit tests for blender."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import numpy as np

from fewshot.data.samplers.blender import MarkovSwitchBlender


class BlenderTests(unittest.TestCase):

  def test_markov_switch(self):
    """Unit test for markov switch blender."""
    ms_blender = MarkovSwitchBlender([0.33, 0.33, 0.34], 0.5, 0)
    x = ms_blender.blend([
        np.zeros([20], dtype=np.int32),
        np.zeros([20], dtype=np.int32),
        np.zeros([20], dtype=np.int32)
    ])
    print(x)


if __name__ == '__main__':
  unittest.main()
