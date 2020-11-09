"""Interface for preprocessijng units.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Preprocessor(object):
  """Preprocessor interface."""

  def __call__(self, inputs):
    return self.preprocess(inputs)

  def preprocess(self, inputs):
    raise NotImplementedError()
