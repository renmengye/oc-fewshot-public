"""Sampler of hierarchical lifelong learning episodes.

It contains multiple sequences of episodes, softly blended together.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
from fewshot.data.samplers.episode_sampler import EpisodeSampler
from fewshot.utils.logger import get as get_logger

log = get_logger()


class HierarchicalEpisodeSampler(EpisodeSampler):

  def __init__(self, subsampler, blender, use_class_hierarchy,
               use_new_class_hierarchy, use_same_family, shuffle_time, seed):
    """Initialize a hiarchical episode sampler.

    Args:
      subsampler: Sampler that samples an individual episode.
      blender: A blender that blends a few sequential episode.
      use_class_hierarchy: Whether to use the class hierarchy defined in the
        dataset.
      use_same_family: Whether to use the same class family across different
        context environment.
      seed: Int. Random seed.
    """
    super(HierarchicalEpisodeSampler, self).__init__(seed)
    self._subsampler = subsampler
    self._blender = blender
    self._use_class_hierarchy = use_class_hierarchy
    self._use_new_class_hierarchy = use_new_class_hierarchy
    self._use_same_family = use_same_family
    if use_same_family:
      log.info('Using the same family across different context.')
    self._shuffle_time = shuffle_time
    self._hierarchy_dict = None
    self._hierarchy_dict_keys = None

  def set_dataset(self, dataset):
    """Hook the sampler with a dataset object."""
    super(HierarchicalEpisodeSampler, self).set_dataset(dataset)
    if self._use_class_hierarchy:
      self._set_hierarchy_dict(dataset.get_hierarchy_dict())

  def _set_hierarchy_dict(self, hierarchy_dict):
    self._hierarchy_dict = hierarchy_dict

  def sample_episode_classes(self,
                             n,
                             nstage=3,
                             max_num=-1,
                             return_flag=False,
                             **kwargs):
    """Samples a sequence of classes relative to 0.

    Args:
      n: Int. Total number of classes.
      nstages: Int. Total number of stages.
      blender: String. Blender type.
      max_num: Int. Maximum number of examples.
      kwargs: Other parameters.
    """
    cls_list = []
    assert np.mod(n, nstage) == 0
    n_ = n // nstage
    if max_num == -1:
      max_num_ = -1
    else:
      assert np.mod(max_num, nstage) == 0
      max_num_ = max_num // nstage
    # Sample each phase individually.
    for phase in range(nstage):
      cls = self.subsampler.sample_episode_classes(
          n_, max_num=max_num_, **kwargs)
      cls_list.append(cls)
    cls_seq, flag = self.blender.blend(
        cls_list, accumulate=not self._use_class_hierarchy)
    if return_flag:
      return cls_seq, flag
    else:
      return cls_seq

  def sample_classes(self, n, save_additional_info=False, **kwargs):
    """Samples a sequence of classes.

    Args:
      n: Int. Number of classes.
      kwargs: Other parameters.
    """

    def compute_stage_gt(stage):
      # print("before", stage)
      # _, stage = np.unique(stage, return_inverse=True)
      _, stage = tf.unique(stage)
      stage = stage.numpy()
      # print("after", stage)
      stage_gt = np.zeros_like(stage)
      cummax = np.maximum.accumulate(stage)
      stage_gt[0] = n
      cond = stage[1:] > cummax[:-1]
      stage_gt[1:] = np.where(cond, n, stage[1:])
      return stage, stage_gt

    def compute_class_count(episode_classes):
      results = []
      counter = {}
      for c in episode_classes:
        if c not in counter:
          counter[c] = 0
        results.append(counter[c])
        counter[c] += 1
      return results

    if not self.use_class_hierarchy:
      assert not self._shuffle_time
      if not save_additional_info:
        return super(HierarchicalEpisodeSampler, self).sample_classes(
            n, **kwargs)
      else:
        episode_classes, stage = self.sample_episode_classes(
            n, return_flag=True, **kwargs)
        classmap = self.cls_dict_keys
        self.rnd.shuffle(classmap)
        results = []
        for c in episode_classes:
          results.append(classmap[c])
        stage, stage_gt = compute_stage_gt(stage)
        info = {
            'stage_id': stage,
            'stage_id_gt': stage_gt,
            'in_stage_class_id': episode_classes,
            'cls_count': np.array(compute_class_count(results))
        }
        return results, info
    else:

      episode_classes, stage = self.sample_episode_classes(
          n, return_flag=True, **kwargs)
      hmap = self.hierarchy_dict_keys
      # H = len(hmap)
      self.rnd.shuffle(hmap)

      if not self.use_new_class_hierarchy:
        # Shuffle the class order within family.
        for s in range(stage.max()):
          self.rnd.shuffle(self.hierarchy_dict[hmap[s]])

        results = []
        for c, s in zip(episode_classes, stage):
          if self._use_same_family:
            N = len(self.cls_dict_keys)
            # print('c', c, 's', s, self.hierarchy_dict[hmap[0]][c] + s * N)
            results.append(self.hierarchy_dict[hmap[0]][c] + s * N)
          else:
            results.append(self.hierarchy_dict[hmap[s]][c])
      else:
        # New type of context here!!!
        # Pick S alphabet, each context, use [idx % i] to pick alphabet within
        # context.
        # Each context is going to have N class. Say 5x25 = 125
        results = []
        # print(episode_classes)
        for c in range(min(episode_classes.max(), len(hmap))):
          self.rnd.shuffle(self.hierarchy_dict[hmap[c]])
        for c, s in zip(episode_classes, stage):
          # Magic number is 3.
          results.append(self.hierarchy_dict[hmap[c % len(hmap)]][s])

      stage, stage_gt = compute_stage_gt(stage)
      if self._shuffle_time:
        idx = np.arange(len(results))
        self.rnd.shuffle(idx)
        results = np.array(results)
        results = results[idx]
        stage = stage[idx]
        stage_gt = stage[idx]
        in_stage_class_id = episode_classes[idx]

      if save_additional_info:
        info = {
            'stage_id': stage,
            'stage_id_gt': stage_gt,
            'in_stage_class_id': episode_classes,
            'cls_count': np.array(compute_class_count(results))
        }
        return results, info
      else:
        return results

  @property
  def subsampler(self):
    """Sampler for each phase."""
    return self._subsampler

  @property
  def blender(self):
    """Blender."""
    return self._blender

  @property
  def use_class_hierarchy(self):
    """Whether or not to use multi-level class hierarchy when sampling classes
    within a stage."""
    return self._use_class_hierarchy

  @property
  def use_new_class_hierarchy(self):
    """Whether or not to use multi-level class hierarchy when sampling classes
    within a stage."""
    return self._use_new_class_hierarchy

  @property
  def hierarchy_dict_keys(self):
    if self._hierarchy_dict_keys is None:
      self._hierarchy_dict_keys = list(self.hierarchy_dict.keys())
    return self._hierarchy_dict_keys

  @property
  def hierarchy_dict(self):
    assert self._hierarchy_dict is not None, 'Uninitialized hierarchy_dict'
    return self._hierarchy_dict
