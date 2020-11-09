"""A multi-stage preprocessor.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fewshot.data.preprocessors.preprocessor import Preprocessor


class SequentialPreprocessor(Preprocessor):
  """A multi-stage preprocessor."""

  def __init__(self, *preprocessors):
    """Preprocessors executed in sequential order."""
    self._preprocessors = preprocessors

  def preprocess(self, inputs):
    for p in self.preprocessors:
      inputs = p(inputs)
    return inputs

  @property
  def preprocessors(self):
    return self._preprocessors
