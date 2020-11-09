"""Sampler interface.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Sampler(object):
  """Basic sampler interface."""

  def sample_collection(self, *args, **kwargs):
    """Samples a collection."""
    raise NotImplementedError()
