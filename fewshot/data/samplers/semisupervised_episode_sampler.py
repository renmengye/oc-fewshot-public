"""A sampler for semi-supervised collections.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.episode_sampler import EpisodeSampler
from fewshot.data.samplers.fewshot_sampler import FewshotSampler
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterSampler('semisupervised')
class SemiSupervisedEpisodeSampler(EpisodeSampler):

  def __init__(self, sampler, seed):
    """Construct a semisupervised sampler.
    Args:
      sampler: Object. A sampler object.
      label_ratio: Float. Proportion of labeled support examples.
      seed: Int. Ranndom seed.
    """
    super(SemiSupervisedEpisodeSampler, self).__init__(seed)
    self._sampler = sampler
    self._distractor_sampler = FewshotSampler(seed)
    self._cls_dict_keys_set = None

  def set_dataset(self, dataset):
    """Hook the sampler with a dataset object."""
    self._set_cls_dict(dataset.get_cls_dict())
    self._sampler.set_dataset(dataset)
    self._distractor_sampler.set_dataset(dataset)

  def sample_classes_exclusion(self, episode_cls, exclusion=None, **kwargs):
    """Samples a sequence of classes, with exclusion.
    Args:
      episode_cls: List. Episodic support set classes.
      exclusion: List. A list of classes excluded from sampling.
    Returns:
      cls: List. A list of bsoslute class IDs.
    """
    classmap = self.cls_dict_keys
    if exclusion is not None:
      exclude_set = set(exclusion)
      remain = list(self.cls_dict_keys_set.difference(exclude_set))
      self.rnd.shuffle(remain)
      return list(map(lambda c: remain[c], episode_cls))
    else:
      self.rnd.shuffle(classmap)
      return list(map(lambda c: classmap[c], episode_cls))

  def sample_label_mask(self, cls_support, label_ratio, const=0.5):
    """See SemiSupervisedEpisodeSampler for documentation."""
    cls_unique, idx, count = np.unique(
        cls_support, return_inverse=True, return_counts=True)
    prob = (1 - label_ratio) * np.exp(-(count - 1) * const) + label_ratio
    prob_all = prob[idx]
    flag = self._rnd.uniform(0.0, 1.0, len(cls_support))
    flag = (flag < prob_all).astype(np.int32)

    # Check if we have missed a class, and ensure it at least appear once.
    for i, c in enumerate(cls_unique):
      appear = np.sum(flag[idx == i])  # Total appearance of a class.
      if appear == 0:  # Total missed
        loc = np.nonzero(idx == i)[0]
        rnd_pick = int(np.floor(self._rnd.uniform(0.0, len(loc))))
        flag[loc[rnd_pick]] = 1
    return flag

  def sample_collection(self,
                        n,
                        m,
                        nd=0,
                        nshotd=0,
                        nqueryd=0,
                        allow_repeat=False,
                        label_ratio=0.5,
                        max_num=-1,
                        save_additional_info=False,
                        **kwargs):
    """Samples an episode of image IDs, plus a flag indicating whether it is
    masked as unlabeled.
    Args:
      n: Int. Number of classes.
      m: Int. Number of query images per class.
      nd: Int. Number of distractor classes.
      nshotd: Int. Number of shot per distractor classes.
      md: Int. Number of query images per distractor class.
      allow_repeat: Bool. Whether allow images to repeat.
    Returns:
      A tuple of
        - a list of support image IDs, and
        - a list of support image label/unlabel flag, and
        - a list of query image IDs, and
        - a list of distractor support image IDs, and
        - a list of distractor query image IDs.
    """
    if save_additional_info:
      results = self._sampler.sample_classes(
          n, max_num=max_num, save_additional_info=True, **kwargs)
      if isinstance(results, tuple):
        cls_support = results[0]
        info = results[1]
      else:
        cls_support = results
        info = {}
    else:
      cls_support = self._sampler.sample_classes(n, max_num=max_num, **kwargs)
    cls_unique = np.unique(np.array(cls_support))
    support = self.sample_images(cls_support, allow_repeat=allow_repeat)
    query = self.sample_test_images(cls_unique, support, m)
    flag = np.array(self.sample_label_mask(cls_support, label_ratio))

    # Sample a bunch of distractor classes here.
    if nd > 0 and nshotd > 0:
      cls = self._distractor_sampler.sample_episode_classes(nd, nshot=nshotd)
      cls_d = np.array(
          self.sample_classes_exclusion(cls, exclusion=cls_unique))
      cls_d_unique = np.unique(cls_d)

      # Assuming dsupport is from a few-shot sampler, shuffle.
      dsupport = np.array(self.sample_images(cls_d, allow_repeat=allow_repeat))
      didx = np.arange(len(cls_d))
      self._rnd.shuffle(didx)
      dsupport = dsupport[didx]
      cls_d = cls_d[didx]

      # Control the maximum length of the support sequence.
      if max_num > 0:
        nremain = max_num - len(support)
        if nremain < len(dsupport):
          dsupport = dsupport[:nremain]
          cls_d = cls_d[:nremain]
      dquery = self.sample_test_images(cls_d_unique, cls_d, nqueryd)

      #  Now mix the distractors and the regular sequence.
      N = len(support)
      N2 = len(dsupport)
      M = len(query)
      M2 = len(dquery)
      is_d = np.concatenate([np.zeros([N]), np.ones([N2])])
      is_d = is_d.astype(np.bool)
      self._rnd.shuffle(is_d)
      not_d = np.logical_not(is_d)
      all_support = np.zeros([N + N2], dtype=np.int32)
      all_support[not_d] = support
      all_support[is_d] = dsupport

      all_support_label = np.zeros([N + N2], dtype=np.int32)
      all_support_label[not_d] = cls_support
      all_support_label[is_d] = cls_d

      all_flag = np.zeros([N + N2], dtype=np.int32)
      all_flag[not_d] = flag
      all_flag[is_d] = 0
      all_query = np.concatenate([query, dquery]).astype(np.int32)

      all_query_label = np.concatenate(
          [np.repeat(cls_unique, m),
           np.repeat(cls_d_unique, nqueryd)]).astype(np.int32)

      collection = {
          'support': all_support,
          'flag': all_flag,
          'query': all_query,
          'support_label': all_support_label,
          'query_label': all_query_label
      }
    else:
      collection = {
          'support': support,
          'support_label': cls_support,
          'flag': flag,
          'query': query,
          'query_label': np.repeat(cls_unique, m)
      }
    if save_additional_info:
      for k in info:
        collection[k] = info[k]
    return collection

  @property
  def cls_dict_keys_set(self):
    """A set of keys in class dictionary."""
    if self._cls_dict_keys_set is None:
      self._cls_dict_keys_set = set(self.cls_dict_keys)
    return self._cls_dict_keys_set
