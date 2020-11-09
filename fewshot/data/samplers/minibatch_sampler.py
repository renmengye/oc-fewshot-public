"""Regular mini-batch sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.sampler import Sampler


@RegisterSampler('minibatch')
class MinibatchSampler(Sampler):
  """Mini-batch sampler."""

  def __init__(self, seed, cycle=True, shuffle=True):
    self._num = None
    self._step = 0
    self._epoch = 0
    self._random = np.random.RandomState(seed)
    self._shuffle_flag = False
    self._cycle = cycle
    self._shuffle = shuffle

  def set_num(self, num):
    assert self._num is None, 'Dataset size can only be set once.'
    self._num = num
    self._shuffle_idx = np.arange(self._num)
    if self._shuffle:
      self._random.shuffle(self._shuffle_idx)

  def reset(self):
    """Reset the iterator."""
    self._shuffle_flag = False
    self._step = 0

  def sample_collection(self, batch_size):
    """See Sampler for documentation."""
    assert self._num is not None, 'Must set dataset size first.'
    # Shuffle data.
    if self._shuffle_flag:
      if self._cycle:
        if self._shuffle:
          self._random.shuffle(self._shuffle_idx)
        self._shuffle_flag = False
        self._epoch += 1
      else:
        return None

    # Calc start/end based on current step.
    start = batch_size * self._step
    end = batch_size * (self._step + 1)
    self._step += 1
    start = start % self._num
    end = end % self._num
    if end > start:
      idx = np.arange(start, end)
      idx = idx.astype(np.int64)
      idx = self._shuffle_idx[idx]
    else:
      if self._cycle:
        idx = np.array(list(range(start, self._num)) + list(range(0, end)))
        idx = idx.astype(np.int64)
        idx = self._shuffle_idx[idx]
      else:
        idx = np.array(list(range(start, self._num)))
      self._shuffle_flag = True
    return idx
