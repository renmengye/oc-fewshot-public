"""Iterator interface.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Iterator(object):
  """General iterator object."""

  def __iter__(self):
    return self

  def next(self):
    raise NotImplementedError()

  def __next__(self):
    return self.next()
