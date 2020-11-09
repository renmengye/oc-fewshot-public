"""A sampler with sequential stages of Chinese Restaurant Process without
reviewing.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.episode_sampler import EpisodeSampler
from fewshot.data.samplers.crp_sampler import CRPSampler


@RegisterSampler('seq_crp')
class SeqCRPSampler(EpisodeSampler):

  def __init__(self, seed):
    super(SeqCRPSampler, self).__init__(seed)
    self._crp_sampler = CRPSampler(seed)

  def sample_episode_classes(self,
                             n,
                             stages=2,
                             alpha=0.5,
                             theta=1.0,
                             max_num=-1,
                             max_num_per_cls=20):
    """See EpisodeSampler class for documentation.

    Args:
      n: Int. Number of classes for each stage.
      stages: Int. Number of sequential stages.
      alpha: Float. Discount parameter.
      theta: Float. Strength parameter.
      max_num: Int. Maximum number of images.
      max_num_per_class: Int. Maximum number of images per class.
    """
    result = []
    cur_max = 0
    assert n % stages == 0
    for i in range(stages):
      result_ = self._crp_sampler.sample_episode_classes(
          n // stages,
          alpha=alpha,
          theta=theta,
          max_num=max_num,
          max_num_per_cls=max_num_per_cls)
      result_ = np.array(result_)
      # print(result_)
      result.extend(list(result_ + cur_max))
      cur_max += result_.max() + 1
    return result
