"""Unit tests for episode iterator.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import unittest

from fewshot.data.datasets.omniglot import OmniglotDataset
from fewshot.data.iterators.episode_iterator import EpisodeIterator
from fewshot.data.preprocessors import NormalizationPreprocessor
from fewshot.data.samplers.fewshot_sampler import FewshotSampler
from fewshot.data.samplers.incremental_sampler import IncrementalSampler


class EpisodeIteratorTests(unittest.TestCase):

  def test_basic(self):
    folder = '/mnt/local/data/omniglot'
    omniglot = OmniglotDataset(folder, 'train')
    preprocessor = NormalizationPreprocessor()
    for bsize in [1, 2]:
      sampler = IncrementalSampler(0)
      it = EpisodeIterator(
          omniglot,
          sampler,
          batch_size=bsize,
          nclasses=5,
          nquery=5,
          expand=True,
          nshot_min=2,
          nshot_max=2,
          preprocessor=preprocessor)
      for x in range(2):
        b = it.next()
        print(b)
        print('support', tf.reduce_max(b.train_images),
              tf.reduce_min(b.train_images), tf.shape(b.train_images))
        print('support label', b.train_labels, tf.shape(b.train_labels))
        print('query', tf.reduce_max(b.test_images),
              tf.reduce_min(b.test_images), tf.shape(b.test_images))
        print('query label', b.test_labels, tf.shape(b.test_labels))

    sampler = FewshotSampler(0)
    it = EpisodeIterator(
        omniglot,
        sampler,
        batch_size=2,
        nclasses=5,
        nquery=5,
        expand=True,
        nshot=2,
        preprocessor=preprocessor)
    for x in range(2):
      b = it.next()
      print(b)
      print('support', tf.reduce_max(b.train_images),
            tf.reduce_min(b.train_images), tf.shape(b.train_images))
      print('support label', b.train_labels, tf.shape(b.train_labels))
      print('query', tf.reduce_max(b.test_images),
            tf.reduce_min(b.test_images), tf.shape(b.test_images))
      print('query label', b.test_labels, tf.shape(b.test_labels))

  def test_fix_unknown(self):
    folder = '/mnt/local/data/omniglot'
    omniglot = OmniglotDataset(folder, 'train')
    preprocessor = NormalizationPreprocessor()
    for bsize in [1, 2]:
      sampler = IncrementalSampler(0)
      it = EpisodeIterator(
          omniglot,
          sampler,
          batch_size=bsize,
          nclasses=5,
          nquery=5,
          expand=True,
          fix_unknown=True,
          nshot_min=2,
          nshot_max=2,
          preprocessor=preprocessor)
      for x in range(2):
        b = it.next()
        print(b)
        print('fixed unknown')
        print('support', tf.reduce_max(b.train_images),
              tf.reduce_min(b.train_images), tf.shape(b.train_images))
        print('support label', b.train_labels, tf.shape(b.train_labels))
        print('support gt', b.train_groundtruth, tf.shape(b.train_groundtruth))
        print('query', tf.reduce_max(b.test_images),
              tf.reduce_min(b.test_images), tf.shape(b.test_images))
        print('query label', b.test_labels, tf.shape(b.test_labels))

    sampler = FewshotSampler(0)
    it = EpisodeIterator(
        omniglot,
        sampler,
        batch_size=2,
        nclasses=5,
        nquery=5,
        expand=True,
        nshot=2,
        fix_unknown=True,
        preprocessor=preprocessor)
    for x in range(2):
      b = it.next()
      print(b)
      print('fixed unknown')
      print('support', tf.reduce_max(b.train_images),
            tf.reduce_min(b.train_images), tf.shape(b.train_images))
      print('support label', b.train_labels, tf.shape(b.train_labels))
      print('support gt', b.train_groundtruth, tf.shape(b.train_groundtruth))
      print('query', tf.reduce_max(b.test_images),
            tf.reduce_min(b.test_images), tf.shape(b.test_images))
      print('query label', b.test_labels, tf.shape(b.test_labels))


if __name__ == '__main__':
  unittest.main()
