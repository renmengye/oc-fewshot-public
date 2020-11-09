"""Unit tests for semi-supervised episode iterator.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import unittest

from fewshot.data.datasets.omniglot import OmniglotDataset
from fewshot.data.iterators.semisupervised_episode_iterator import SemiSupervisedEpisodeIterator  # NOQA
from fewshot.data.preprocessors import NormalizationPreprocessor
from fewshot.data.samplers.crp_sampler import CRPSampler
from fewshot.data.samplers.semisupervised_episode_sampler import SemiSupervisedEpisodeSampler  # NOQA


class SemiSupervisedEpisodeIteratorTests(unittest.TestCase):

  def test_basic(self):
    folder = '/mnt/local/data/omniglot'
    omniglot = OmniglotDataset(folder, 'train')
    preprocessor = NormalizationPreprocessor()
    for bsize in [1, 2]:
      sampler = CRPSampler(0)
      sampler2 = SemiSupervisedEpisodeSampler(sampler, 0)
      it = SemiSupervisedEpisodeIterator(
          omniglot,
          sampler2,
          batch_size=bsize,
          nclasses=10,
          nquery=5,
          preprocessor=preprocessor,
          expand=True,
          fix_unknown=True,
          label_ratio=0.5,
          nd=5,
          sd=1,
          md=2,
          alpha=0.5,
          theta=1.0)
      for x in range(2):
        b = it.next()
        print(b)
        print('support', tf.reduce_max(b.train_images),
              tf.reduce_min(b.train_images), tf.shape(b.train_images))
        print('support label', b.train_labels, tf.shape(b.train_labels))
        print('support gt', b.train_groundtruth, tf.shape(b.train_groundtruth))
        print('query', tf.reduce_max(b.test_images),
              tf.reduce_min(b.test_images), tf.shape(b.test_images))
        print('query label', b.test_labels, tf.shape(b.test_labels))


if __name__ == '__main__':
  unittest.main()
