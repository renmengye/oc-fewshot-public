"""Blending lists of classes, used for hierarchical samplers.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

blender_dict = {}


def register_blender(name):

  def decorator(f):
    blender_dict[name] = f
    return f

  return decorator


def get_blender(name, **kwargs):
  return blender_dict[name](**kwargs)


class Blender(object):

  def blend(self, cls_list):
    """Returns the blended class list and the hierarchical flag."""
    raise NotImplementedError


@register_blender('hard')
class HardBlender(Blender):
  """Simple concatenation of classes."""

  def blend(self, cls_list, accumulate=True):
    result = []
    counter = 0
    flag = []
    for j, cls in enumerate(cls_list):
      if accumulate:
        result.extend([c + counter for c in cls])
        counter += len(cls)
      else:
        result.extend(cls)
      flag.extend([j] * len(cls))
    return np.array(result), np.array(flag)


@register_blender('blur')
class BlurBlender(Blender):
  """Concatenation with some soft mixing using convolutional blurring
    operation.
  """

  def __init__(self, window_size, stride, nrun, seed):
    """Initialize a blur blender.

    Args:
      window_size: Window size to blur.
      stride: Stride size.
      nrun: Number of blurring operations.
      seed: Random seed.
    """
    self._wsize = window_size
    self._stride = stride
    self._nrun = nrun
    self._rnd = np.random.RandomState(seed)

  def blend(self, cls_list, accumulate=True):
    result = []
    flag = []
    counter = 0
    # Add absolute class ID.
    for j, cls in enumerate(cls_list):
      if accumulate:
        result.extend([c + counter for c in cls])
        counter += (max(cls) + 1)
      else:
        result.extend(cls)
      flag.extend([j] * len(cls))

    result = np.array(result)
    flag = np.array(flag)
    wsize = 10
    stride = 3
    arr = np.arange(len(result))
    for j in range(self._nrun):
      for i in range(0, len(result), self._stride):
        if i + self._wsize > len(result):
          break
        subwindow = arr[i:i + self._wsize]
        self._rnd.shuffle(subwindow)
        arr[i:i + self._wsize] = subwindow

    result = result[arr]
    flag = flag[arr]
    return result, flag


@register_blender('markov-switch')
class MarkovSwitchBlender(Blender):
  """Using a markov switching process."""

  def __init__(self, base_dist, switch_prob, seed):
    """Initialize a markov switch blender.

    Args:
      base_dist: List. Base distribution shared by subsamplers.
      switch_prob: Probability to switch stage.
      seed: Int. Random seed.
    """
    self._base_dist = base_dist
    self._switch_prob = switch_prob
    self._rnd = np.random.RandomState(seed)

  def sample_random(self):
    """Randomly sample an integer from 0 to len(base_dist) - 1."""
    return np.argmax(self._rnd.multinomial(1, self._base_dist, size=1))

  def sample_switch(self):
    """Sample a switch bool variable."""
    return self._rnd.uniform(0.0, 1.0) < self._switch_prob

  def blend(self, cls_list, accumulate=True):
    """Blend a sequence of classes."""
    totallen = sum([len(c) for c in cls_list])
    cls_list2 = [None] * len(cls_list)
    counter = 0

    s = 0
    counter = 0
    result = []
    flag = []

    # Markov switching process.
    while len(result) < totallen:

      # Reindex the classes.
      if cls_list2[s] is None:

        # Add class absolute ID.
        if accumulate:
          cls_list2[s] = [c + counter for c in cls_list[s]]
          counter += (max(cls_list[s]) + 1)
        else:
          cls_list2[s] = [c for c in cls_list[s]]

      # Pop from the list if there is still remaining.
      if len(cls_list2[s]) > 0:
        result.append(cls_list2[s].pop(0))
        flag.append(s)
      else:
        s = self.sample_random()
        continue

      # Decides whether to switch stage.
      if self.sample_switch():
        s = self.sample_random()

    return np.array(result), np.array(flag)
