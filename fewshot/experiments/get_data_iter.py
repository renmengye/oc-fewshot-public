"""
Build data iterators.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.data_factory import get_sampler
from fewshot.data.iterators import EpisodeIterator
from fewshot.data.iterators import MinibatchIterator
from fewshot.data.iterators import SemiSupervisedEpisodeIterator
from fewshot.data.iterators import SimEpisodeIterator
from fewshot.data.preprocessors import DataAugmentationPreprocessor
from fewshot.data.preprocessors import NCHWPreprocessor
from fewshot.data.preprocessors import NormalizationPreprocessor
from fewshot.data.preprocessors import SequentialPreprocessor
from fewshot.data.preprocessors import RandomBoxOccluder
from fewshot.data.samplers import FewshotSampler
from fewshot.data.samplers import HierarchicalEpisodeSampler
from fewshot.data.samplers import MinibatchSampler
from fewshot.data.samplers import MixSampler
from fewshot.data.samplers import SemiSupervisedEpisodeSampler
from fewshot.data.datasets import UppsalaDataset
from fewshot.data.episode_processors import BackgroundProcessor
from fewshot.data.episode_processors import BackgroundProcessorV2
from fewshot.data.samplers.blender import get_blender


def get_dataiter(data,
                 batch_size,
                 nchw=False,
                 data_aug=False,
                 distributed=False):
  """Gets dataset iterator."""
  md = data['metadata']
  if distributed:
    import horovod.tensorflow as hvd
    rank = hvd.rank()
    seed = rank * 1234
  else:
    seed = 0
  sampler_dict = {}
  sampler_dict['train'] = MinibatchSampler(seed, cycle=True, shuffle=True)
  sampler_dict['val'] = MinibatchSampler(seed, cycle=False, shuffle=False)
  sampler_dict['test'] = MinibatchSampler(seed, cycle=False, shuffle=False)
  norm_prep = NormalizationPreprocessor(
      mean=np.array(md.mean_pix), std=np.array(md.std_pix))
  da_prep = DataAugmentationPreprocessor(md.image_size, md.crop_size,
                                         md.random_crop, md.random_flip,
                                         md.random_color, md.random_rotate)
  if nchw:
    nchw_prep = NCHWPreprocessor()
    if data_aug:
      train_prep = SequentialPreprocessor(da_prep, norm_prep, nchw_prep)
    else:
      train_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    val_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    test_prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    train_prep = SequentialPreprocessor(da_prep, norm_prep)
    val_prep = norm_prep
    test_prep = norm_prep
  prep = {'train': train_prep, 'val': val_prep, 'test': test_prep}
  it_dict = {}
  for k in ['train', 'val', 'test']:
    cycle = k == 'train'
    it_dict[k] = MinibatchIterator(data[k], sampler_dict[k], batch_size,
                                   prep[k])
  return it_dict


def get_dataiter_fewshot(data,
                         data_config,
                         batch_size=1,
                         nchw=False,
                         prefetch=True,
                         distributed=False):
  """Gets few-shot episode iterator."""
  md = data['metadata']
  if distributed:
    import horovod.tensorflow as hvd
    rank = hvd.rank()
    seed = rank * 1234
  else:
    seed = 0
  sampler_dict = {}
  sampler_dict['train_fs'] = FewshotSampler(seed)
  sampler_dict['val_fs'] = FewshotSampler(seed)
  sampler_dict['test_fs'] = FewshotSampler(seed)
  norm_prep = NormalizationPreprocessor(
      mean=np.array(md.mean_pix), std=np.array(md.std_pix))
  if nchw:
    nchw_prep = NCHWPreprocessor()
    prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    prep = norm_prep
  it_dict = {}
  # For evaluation only. No additional preprocessor.
  for k in ['train_fs', 'val_fs', 'test_fs']:
    if data[k] is not None:
      it_dict[k] = EpisodeIterator(
          data[k],
          sampler_dict[k],
          batch_size=batch_size,
          nclasses=data_config.nway,
          nquery=data_config.nquery,
          nshot=data_config.nshot_max,
          preprocessor=prep,
          prefetch=prefetch,
          maxlen=data_config.nshot_max * data_config.nway)
    else:
      it_dict[k] = None
  return it_dict


def get_dataiter_continual(data,
                           data_config,
                           batch_size=1,
                           nchw=True,
                           prefetch=True,
                           save_additional_info=False,
                           random_box=False,
                           distributed=False,
                           seed=0):
  """Gets few-shot episode iterator.

  Args:
    data: Object. Dataset object.
    data_config: Config. Data source episode config.
    batch_size: Int. Batch size.
    nchw: Bool. Whether to transpose the images to NCHW.
    prefetch: Bool. Whether to add prefetching module in the data loader.
    save_additional_info: Bool. Whether to add additional episodic information.
  """
  md = data['metadata']
  data['trainval_fs'] = data['train_fs']  # Use the same for trainval.
  if data_config.base_sampler == 'incremental':
    kwargs = {
        'nshot_min': 1,
        'nshot_max': data_config.nshot_max,
        'allow_repeat': data_config.allow_repeat
    }
  elif data_config.base_sampler == 'fewshot':
    kwargs = {
        'nshot_min': data_config.nshot_max,
        'nshot_max': data_config.nshot_max,
        'allow_repeat': data_config.allow_repeat
    }
  elif data_config.base_sampler == 'constant_prob':
    kwargs = {
        'p': data_config.prob_new,
        'allow_repeat': data_config.allow_repeat,
        'max_num': data_config.maxlen,
        'max_num_per_cls': data_config.max_num_per_cls
    }
  elif data_config.base_sampler == 'crp':
    kwargs = {
        'alpha': data_config.crp_alpha,
        'theta': data_config.crp_theta,
        'allow_repeat': data_config.allow_repeat,
        'max_num': data_config.maxlen,
        'max_num_per_cls': data_config.max_num_per_cls
    }
  elif data_config.base_sampler == 'seq_crp':
    kwargs = {
        'alpha': data_config.crp_alpha,
        'theta': data_config.crp_theta,
        'allow_repeat': data_config.allow_repeat,
        'max_num': data_config.maxlen // 2,
        'max_num_per_cls': data_config.max_num_per_cls,
        'stages': 2
    }
  else:
    raise ValueError('Not supported')

  # Split.
  split_list = ['train_fs', 'trainval_fs', 'val_fs', 'test_fs']
  # Gets episodic sampler.
  sampler_dict = {}

  seed_list = [seed, 1001, 0, 0]
  if distributed:
    import horovod.tensorflow as hvd
    seed2 = hvd.rank() * 1234 + np.array(seed_list)
  else:
    seed2 = seed_list
  for k, s in zip(split_list, seed2):
    sampler_dict[k] = get_sampler(data_config.base_sampler, s)

  # Wrap it with hierarchical sampler.
  if data_config.hierarchical:
    for k, s in zip(split_list, seed2):
      if data_config.blender in ['hard']:
        blender = get_blender(data_config.blender)
      elif data_config.blender in ['blur']:
        blender = get_blender(
            data_config.blender,
            window_size=data_config.blur_window_size,
            stride=data_config.blur_stride,
            nrun=data_config.blur_nrun,
            seed=s)
      elif data_config.blender in ['markov-switch']:
        blender = get_blender(
            data_config.blender,
            base_dist=np.ones([data_config.nstage]) / float(
                data_config.nstage),
            switch_prob=data_config.markov_switch_prob,
            seed=s)
      else:
        raise ValueError('Unknown blender {}'.format(data_config.blender))

      # Mix class hierarchy and non class hierarchy.
      if data_config.mix_class_hierarchy:
        sampler_dict[k] = MixSampler([
            HierarchicalEpisodeSampler(sampler_dict[k], blender, False,
                                       data_config.use_new_class_hierarchy,
                                       data_config.use_same_family,
                                       data_config.shuffle_time, s),
            HierarchicalEpisodeSampler(sampler_dict[k], blender, True,
                                       data_config.use_new_class_hierarchy,
                                       data_config.use_same_family,
                                       data_config.shuffle_time, s + 1)
        ], [0.5, 0.5], 1023)  # Set for 0.5/0.5 for now.
      else:
        sampler_dict[k] = HierarchicalEpisodeSampler(
            sampler_dict[k], blender, data_config.use_class_hierarchy,
            data_config.use_new_class_hierarchy, data_config.use_same_family,
            data_config.shuffle_time, s)
      kwargs['nstage'] = data_config.nstage

  # Wrap it with semisupervised sampler.
  if data_config.semisupervised:
    for k, s in zip(split_list, seed2):
      sampler_dict[k] = SemiSupervisedEpisodeSampler(sampler_dict[k], s)
      kwargs['label_ratio'] = data_config.label_ratio
      kwargs['nd'] = data_config.distractor_nway
      kwargs['nshotd'] = data_config.distractor_nshot
      kwargs['md'] = data_config.distractor_nquery

  # Random background.
  if data_config.random_background != 'none':
    if data_config.random_background in ['uppsala', 'uppsala_double']:
      folder = './data/uppsala-texture'
      bg_dataset = [
          UppsalaDataset(folder, s) for s in ['train', 'train', 'val', 'test']
      ]
    else:
      assert False

    if data_config.random_background == 'uppsala':
      Processor = BackgroundProcessor
    elif data_config.random_background == 'uppsala_double':
      Processor = BackgroundProcessorV2
    bg_random = [True, False, False, False]
    bg_processor_dict = dict(
        zip(split_list, [
            Processor(
                d,
                random=r,
                random_apply=data_config.random_background_random_apply,
                apply_prob=data_config.random_background_apply_prob,
                random_context=data_config.random_background_random_context,
                gaussian_noise_std=data_config.random_background_gaussian_std)
            for d, r in zip(bg_dataset, bg_random)
        ]))
  else:
    bg_processor_dict = dict(zip(split_list, [None] * len(split_list)))

  # Data preprocessors.
  norm_prep = NormalizationPreprocessor(
      mean=np.array(md.mean_pix), std=np.array(md.std_pix))
  da_prep = DataAugmentationPreprocessor(md.image_size, md.crop_size,
                                         md.random_crop, md.random_flip,
                                         md.random_color, md.random_rotate)
  nchw_prep = NCHWPreprocessor()
  random_box_prep = RandomBoxOccluder()

  train_prep_list = [da_prep]
  val_prep_list = []
  if random_box:
    train_prep_list.append(random_box_prep)
    val_prep_list.append(random_box_prep)
  train_prep_list.append(norm_prep)
  val_prep_list.append(norm_prep)
  if nchw:
    train_prep_list.append(nchw_prep)
    val_prep_list.append(nchw_prep)
  train_prep = SequentialPreprocessor(*train_prep_list)
  val_prep = SequentialPreprocessor(*val_prep_list)

  prep = {
      'train_fs': train_prep,
      'trainval_fs': val_prep,
      'val_fs': val_prep,
      'test_fs': val_prep
  }

  # Adds data iterators.
  if data_config.semisupervised:
    IteratorClass = SemiSupervisedEpisodeIterator
  else:
    IteratorClass = EpisodeIterator

  it_dict = {}

  # Key, batch size
  for k, b in zip(['train_fs', 'trainval_fs', 'val_fs', 'test_fs'],
                  [batch_size, 1, 1, 1]):
    it_dict[k] = IteratorClass(
        data[k],
        sampler_dict[k],
        batch_size=b,
        nclasses=data_config.nway,
        nquery=data_config.nquery,
        preprocessor=prep[k],
        episode_processor=bg_processor_dict[k],
        maxlen=data_config.maxlen,
        fix_unknown=data_config.fix_unknown,
        prefetch=prefetch,
        save_additional_info=save_additional_info,
        **kwargs)
  return it_dict


def get_dataiter_sim(data,
                     data_config,
                     batch_size=1,
                     nchw=True,
                     prefetch=True,
                     distributed=False,
                     seed=0):
  """Gets few-shot episode iterator in simulated environment.

  Args:
    data: Object. Dataset object.
    data_config: Config. Data source episode config.
    batch_size: Int. Batch size.
    nchw: Bool. Whether to transpose the images to NCHW.
    prefetch: Bool. Whether to add prefetching module in the data loader.
  """
  md = data['metadata']
  data['trainval_fs'] = data['train_fs']  # Use the same for trainval.

  # Split.
  split_list = ['train_fs', 'trainval_fs', 'val_fs', 'test_fs']

  # Gets episodic sampler.
  sampler_dict = {}

  seed_list = [seed, 1001, 0, 0]
  if distributed:
    import horovod.tensorflow as hvd
    seed2 = hvd.rank() * 1234 + np.array(seed_list)
  else:
    seed2 = seed_list

  cycle_list = [True, False, False, False]
  shuffle_list = [True, False, False, False]

  for k, s, cyc, shuf in zip(split_list, seed2, cycle_list, shuffle_list):
    sampler_dict[k] = MinibatchSampler(s, cycle=cyc, shuffle=shuf)

  # Data preprocessors.
  norm_prep = NormalizationPreprocessor(
      mean=np.array(md.mean_pix), std=np.array(md.std_pix))

  # Adds data iterators.
  IteratorClass = SimEpisodeIterator

  it_dict = {}

  # Key, batch size
  for k, b, r, s in zip(['train_fs', 'trainval_fs', 'val_fs', 'test_fs'],
                        [batch_size, 1, 1, 1], [True, False, False, False],
                        seed2):
    it_dict[k] = IteratorClass(
        data[k],
        sampler_dict[k],
        batch_size=b,
        nclasses=data_config.nway,
        preprocessor=norm_prep,
        maxlen=data_config.maxlen,
        fix_unknown=data_config.fix_unknown,
        semisupervised=data_config.semisupervised,
        label_ratio=data_config.label_ratio,
        prefetch=prefetch,
        random_crop=r,
        random_drop=r,
        random_flip=r,
        random_jitter=r,
        max_num_per_cls=data_config.max_num_per_cls,
        seed=s)
  return it_dict
