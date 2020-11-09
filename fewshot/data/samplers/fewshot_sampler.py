"""Regular few-shot episode sampler.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.incremental_sampler import IncrementalSampler


@RegisterSampler('fewshot')
class FewshotSampler(IncrementalSampler):
  """Standard few-shot learning sampler."""

  def __init__(self, seed):
    super(FewshotSampler, self).__init__(seed)

  def sample_episode_classes(self, n, nshot=1, **kwargs):
    """See EpisodeSampler for documentation."""
    return super(FewshotSampler, self).sample_episode_classes(
        n, nshot_min=nshot, nshot_max=nshot)
