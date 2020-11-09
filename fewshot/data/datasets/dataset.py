"""Dataset interface.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Dataset(object):
  """A general random access dataset."""

  def __init__(self):
    pass

  def get_images(self, inds):
    """Gets images based on indices."""
    raise NotImplementedError()

  def get_labels(self, inds):
    """Gets labels based on indices."""
    raise NotImplementedError()

  def get_size(self):
    """Gets the size of the dataset."""
    raise NotImplementedError()
