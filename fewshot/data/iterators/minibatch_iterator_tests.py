"""Unit tests for mini-batch iterator.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import unittest

from fewshot.data.samplers.minibatch_sampler import MinibatchSampler
from fewshot.data.iterators.minibatch_iterator import MinibatchIterator
from fewshot.data.datasets.omniglot import OmniglotDataset
from fewshot.data.preprocessors import NormalizationPreprocessor


class MinibatchIteratorTests(unittest.TestCase):

  def test_basic(self):
    # TODO: change this to a fake dataset instance.
    folder = '/mnt/local/data/omniglot'
    omniglot = OmniglotDataset(folder, 'train')
    preprocessor = NormalizationPreprocessor()
    sampler = MinibatchSampler(0)
    it = MinibatchIterator(omniglot, sampler, preprocessor=preprocessor)
    for x in range(2):
      b = it.next(128)
      print(b)
      print('support', tf.reduce_max(b.train_images),
            tf.reduce_min(b.train_images), tf.shape(b.train_images))
      print('support label', b.train_labels, tf.shape(b.train_labels))


if __name__ == '__main__':
  unittest.main()
