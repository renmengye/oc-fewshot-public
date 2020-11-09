"""Random-shot sampler in the regular few-shot setting.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import absolute_import, division, unicode_literals

import numpy as np

from fewshot.data.data_factory import RegisterSampler
from fewshot.data.samplers.episode_sampler import EpisodeSampler


@RegisterSampler('randomshot')
class RandomshotSampler(EpisodeSampler):
  """Samples an episode of random shot. Note that within a single episode, the
  shot number is the same."""

  def __init__(self, seed):
    super(RandomshotSampler, self).__init__(seed)
    self._cls_dict_keys = None

  def sample_classes(self, n, nshot_min=1, nshot_max=1, **kwargs):
    """See Sampler class for ducumentation."""
    assert nshot_min >= 1, 'At least 1 images.'
    assert nshot_max >= nshot_min, 'Max >= min.'
    nimages = self.rnd.uniform(nshot_min, nshot_max + 1.0, [1])
    nimages = np.floor(nimages).astype(int)
    classmap = self.cls_dict_keys
    self.rnd.shuffle(classmap)
    results = []
    for i in range(n):
      results.extend([classmap[i]] * nimages[0])
    return results

  @property
  def cls_dict_keys(self):
    if self._cls_dict_keys is None:
      self._cls_dict_keys = list(self.cls_dict.keys())
    return self._cls_dict_keys
