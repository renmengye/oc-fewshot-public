"""Incremental class episode sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.episode_sampler import EpisodeSampler


@RegisterSampler('incremental')
class IncrementalSampler(EpisodeSampler):
  """Samples classes incrementally without reviewing.
  Using uniform distribution of the number of images per class.
  """

  def sample_episode_classes(self,
                             n,
                             nshot_min=1,
                             nshot_max=1,
                             maxnum=None,
                             **kwargs):
    """See EpisodeSampler class for documentation."""
    assert nshot_min >= 1, 'At least 1 images.'
    assert nshot_max >= nshot_min, 'Max >= min.'
    nimages = self.rnd.uniform(nshot_min, nshot_max + 1.0, [n])
    nimages = np.floor(nimages).astype(int)
    results = []
    for i, m in enumerate(nimages):
      results.extend([i] * m)
    return results
