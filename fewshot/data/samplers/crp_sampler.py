"""A sampler with Chinese Restaurant Process.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.episode_sampler import EpisodeSampler


@RegisterSampler('crp')
class CRPSampler(EpisodeSampler):

  def sample_episode_classes(self,
                             n,
                             alpha=0.5,
                             theta=1.0,
                             max_num=-1,
                             max_num_per_cls=20,
                             **kwargs):
    """See EpisodeSampler class for documentation.

    Args:
      n: Int. Number of classes.
      alpha: Float. Discount parameter.
      theta: Float. Strength parameter.
      max_num: Int. Maximum number of images.
      max_num_per_class: Int. Maximum number of images per class.
    """
    k = 0  # Current maximum table count.
    c = 0  # Current sampled table.
    m = 0  # Current total number of people.
    m_array = np.zeros([n])
    result = []
    # We need n tables in total.
    # print('alpha', alpha)
    # print('max num', max_num)
    # print('max num', max_num)
    # print('max num pc', max_num_per_cls)
    # print(max_num_per_cls)
    while k <= n and (max_num < 0 or m < max_num):
      # print('k', k, 'm', m)
      p_new = (theta + k * alpha) / (m + theta)
      if k == n:
        p_new = 0.0
      # print('pnew', p_new)
      p_old = (m_array[:k] - alpha) / (m + theta)
      # print('pold', p_old)
      pvals = list(p_old) + [p_new]
      sample = self.rnd.multinomial(1, pvals, size=1)
      sample = sample.reshape([-1])
      # print('sample', sample, sample.shape)
      if k == n:  # Reached the maximum classes.
        if sample[-1] == 1:  # Just sampled one more.
          break
          # continue

      # print('sample', sample, sample.shape)
      idx = np.argmax(sample)
      if m_array[idx] < max_num_per_cls:
        m_array[idx] += 1
      else:
        continue  # Cannot sample more on this table.
      k = np.sum(m_array > 0)  # Update total table.
      m += 1  # Update total guests.
      result.append(idx)
      # print(result)
    # print(result)
    # print('marray', m_array, 'results', result)
    return result
