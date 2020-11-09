"""A sampler with constant probability of reviewing old and new.

Author: Mengye Ren (mren@cs.toronto.edu)

Note:
- Prob. of seeing new class is p, (except for class-0).
- Prob. of seeing old class is (1-p).
- Prob. of seeing a particular old class is (1/p) / K, where K=number of old
  classes seen so far.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.episode_sampler import EpisodeSampler


@RegisterSampler('constant_prob')
class ConstantProbSampler(EpisodeSampler):

  def sample_episode_classes(self, n, p=0.1, max_num=-1, max_num_per_cls=20):
    """See EpisodeSampler class for documentation."""
    k = 0  # Current maximum class.
    c = 0  # Sampled class.
    result = [0]
    m_array = np.zeros([n + 1])
    m_array[0] = 1
    while k < n and (max_num < 0 or len(result) < max_num):
      success = False
      assert max_num_per_cls == 20
      while not success:
        r = self.rnd.uniform(0.0, 1.0)
        if r >= 1 - p:
          # Add a new class.
          k = k + 1
          c = k
        else:
          # Take one of the old class, at random.
          c = int(np.floor(r / (1 - p) * (k + 1)))
        if m_array[c] < max_num_per_cls or max_num_per_cls < 0:
          success = True
          m_array[c] += 1
      result.append(c)
    if result[-1] >= n:
      result = result[:-1]
    return result
