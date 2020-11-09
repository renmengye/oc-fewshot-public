"""Module interface.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Module(object):

  def __call__(self, *args, **kwargs):
    """Extract features from raw inputs.

    Args:
      x: [N, ...]. Inputs.
      is_training: Bool. Whether in training mode.
    """
    return self.forward(*args, **kwargs)

  def forward(self, *args, **kwargs):
    """Extract features from raw inputs.

    Args:
      x: [N, ...]. Inputs.
      is_training: Bool. Whether in training mode.
    """
    raise NotImplementedError()

  def _prefix(self, prefix, name):
    if prefix is None:
      return name
    else:
      return prefix + '/' + name
