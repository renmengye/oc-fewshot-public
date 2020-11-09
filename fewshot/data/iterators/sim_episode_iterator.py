"""Simulation few-shot episode iterator.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.iterators.iterator import Iterator
from fewshot.data.registry import RegisterIterator
from fewshot.utils.logger import get as get_logger

log = get_logger()


def transform_bbox(bboxes, new_dim, original_dim=(600, 800)):
  bboxes = bboxes.reshape([-1, 2, 2]) * new_dim / np.array(original_dim)
  return bboxes.reshape([-1, 4]).astype(np.int64)


@RegisterIterator('sim_episode')
class SimEpisodeIterator(Iterator):

  def __init__(self,
               dataset,
               sampler,
               batch_size,
               nclasses,
               preprocessor=None,
               fix_unknown=False,
               maxlen=-1,
               semisupervised=False,
               label_ratio=0.1,
               prefetch=True,
               random_crop=False,
               random_shuffle_objects=True,
               max_num_per_cls=6,
               max_bbox_per_object=9,
               transform_bbox=True,
               random_drop=False,
               random_flip=False,
               random_jitter=False,
               seed=0):
    """Creates a lifelong learning data iterator based on simulation data.

    Args:
      dataset: A dataset object. See `fewshot/data/dataset.py`.
      sampler: A sampler object. See `fewshot/data/sampler.py`.
      batch_size: Number of episodes together.
      fix_unknown: Whether the unknown token is k+1 or fixed at K+1.
      maxlen: Maximum length of the sequence.
      preprocessor: Image preprocessor.
    """
    self._dataset = dataset
    self._sampler = sampler
    self._max_bbox_per_object = max_bbox_per_object
    self._transform_bbox = transform_bbox
    sampler.set_num(dataset.get_size())
    self._batch_size = batch_size
    self._nclasses = nclasses
    self._random_crop = random_crop
    self._random_drop = random_drop
    self._random_flip = random_flip
    self._random_jitter = random_jitter
    self._random_shuffle_objects = random_shuffle_objects
    self._max_num_per_cls = max_num_per_cls
    self._rnd = np.random.RandomState(seed)
    self._preprocessor = preprocessor
    self._maxlen = maxlen
    self._prefetch = prefetch
    self._semisupervised = semisupervised
    self._label_ratio = label_ratio
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)
    assert fix_unknown, "Not supported"

  def sample_label_mask(self, cls_support, label_ratio, const=0.5):
    """See SemiSupervisedEpisodeSamplerV2 for documentation."""
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

  def pad_x(self, x, maxlen):
    """Pad image sequence."""
    T = x.shape[0]
    return np.pad(x, [[0, maxlen - T], [0, 0], [0, 0], [0, 0]], mode='reflect')

  def pad_y(self, y, maxlen):
    """Pad label sequence."""
    T = y.shape[0]
    return np.pad(y, [0, maxlen - T], mode='constant', constant_values=0)

  def pad_z(self, y, maxlen):
    diff = maxlen - len(y)
    idxs = list(range(len(y)))[::-1]
    padded_idxs = np.pad(idxs, [0, diff], mode='reflect')[1:]
    pad_array = y[padded_idxs]
    padding = pad_array[range(diff)]
    y_padded = np.array(y.tolist() + padding.tolist()).astype(y.dtype)
    return y_padded

  def process_one(self, idx):
    """Process episodes.

    Args:
      idx: An integer ID.
    """
    # print('idx', idx)
    raw = self.dataset.get_episode(idx)
    H = raw[0]['rgb'].shape[0]
    W = raw[0]['rgb'].shape[1]
    T = self.maxlen
    fcount = 0  # Frame counter
    ocount = 0  # Object counter
    unk_id = self.nclasses
    # 4 channel image. The last channel is the object segmentation mask.
    # Allocate more than necessary first.
    N = sum([len(item['annotation'].keys()) for item in raw])
    N = max(N, T)
    x_s = np.zeros([N, H, W, 3], dtype=np.uint8)
    x_att = np.zeros([N, H, W, 1], dtype=np.uint8)
    y_s = np.zeros([N], dtype=np.int32)
    flag = np.zeros([N], dtype=np.int32)
    label_dict = {}
    categories = np.zeros([N], dtype='|S15')
    instance_ids = np.zeros([N], dtype='|S15')
    bboxes = np.zeros([N, self._max_bbox_per_object, 4], dtype=np.int32)
    label_dicts = []
    annotations = []

    # Each frame has a centric object
    for item in raw:
      # Shuffle the objects in the frame.
      keys = list(item['annotation'].keys())
      if len(keys) > 0:
        if self._random_shuffle_objects:
          self._rnd.shuffle(keys)
        for key in keys:

          # Skip black images.
          if np.sum(item['rgb']) == 0:
            # log.info('Black out found')
            # assert False, 'Found black image'
            continue

          x_s[fcount] = item['rgb']
          obj = item['annotation'][key]
          inst_id = obj['instance_id']
          annotation = item['annotation']
          x_s[fcount] = item['rgb']
          obj = annotation[key]
          inst_id = obj['instance_id']

          category = np.array(obj['category'])
          instance_id = np.array(obj['instance_id'])
          bbox_i = np.array(obj['zoom_bboxes'])

          # Skip because it exeeds the total number of classes.
          if inst_id not in label_dict and ocount == self.nclasses - 1:
            continue

          if inst_id not in label_dict:
            label_dict[inst_id] = ocount
            ocount += 1

          attention_map = (item['instance_seg'] == inst_id).astype(np.uint8)

          if self._transform_bbox:
            bbox_i = transform_bbox(bbox_i, attention_map.shape[:2])

          if np.all(attention_map == 0):
            attention_bbox = bbox_i[-1]
            # print(attention_bbox)

            if np.all(attention_bbox == 0):
              log.error("Bbox empty!")
              assert False

            attention_map = np.zeros_like(attention_map)
            y1, y2, x1, x2 = attention_bbox
            attention_map[y1:y2, x1:x2] = 1.0

          x_att[fcount, :, :, 0] = attention_map

          if self._random_jitter:
            PY = 6
            PX = 8
            x_pad_ = np.pad(
                x_s[fcount], [[PY, PY], [PX, PX], [0, 0]],
                mode='constant',
                constant_values=0)
            x_att_pad_ = np.pad(
                x_att[fcount], [[PY, PY], [PX, PX], [0, 0]],
                mode='constant',
                constant_values=0)
            H = x_s[fcount].shape[0]
            W = x_s[fcount].shape[1]

            # Jitter image and segmentation differently.
            start_y = self._rnd.randint(0, PY * 2)
            start_x = self._rnd.randint(0, PX * 2)
            start_y2 = self._rnd.randint(max(0, start_y - 2), start_y + 2)
            start_x2 = self._rnd.randint(max(0, start_x - 2), start_x + 2)
            x_s[fcount] = x_pad_[start_y:start_y + H, start_x:start_x + W]
            x_att[fcount] = x_att_pad_[start_y2:start_y2 +
                                       H, start_x2:start_x2 + W]

          y_s[fcount] = label_dict[inst_id]
          flag[fcount] = 1

          n_bboxes = bbox_i.shape[0]
          bboxes[fcount, :n_bboxes] = bbox_i

          categories[fcount] = category
          instance_ids[fcount] = instance_id
          fcount += 1

    # Drop some objects that has too frequent appearances
    MAX = self._max_num_per_cls
    ys_unique, ys_counts = np.unique(y_s, return_counts=True)
    oversize_idx = ys_unique[ys_counts > MAX]
    oversize_count = ys_counts[ys_counts > MAX]
    prob = np.ones([len(y_s)])

    for idx, cnt in zip(oversize_idx, oversize_count):
      prob[y_s == idx] = MAX / float(cnt)

    # Random drop 0.95 uniform.
    if self._random_drop:
      prob *= 0.95

    keep = self._rnd.uniform(0.0, 1.0, len(y_s)) < prob
    x_s = x_s[keep]
    x_att = x_att[keep]
    y_s = y_s[keep]
    bboxes = bboxes[keep]
    categories = categories[keep]
    instance_ids = instance_ids[keep]
    flag = flag[keep]
    fcount = x_s.shape[0]

    if fcount == 0:
      return None

    if fcount < T:
      x_s = self.pad_x(x_s, T)
      x_att = self.pad_x(x_att, T)
      y_s = self.pad_y(y_s, T)
      flag = self.pad_y(flag, T)

      bboxes = self.pad_z(bboxes, T)
      categories = self.pad_z(categories, T)
      instance_ids = self.pad_z(instance_ids, T)

    # Crop into T frames.
    if fcount > T:
      # random sample T frames.
      if self._random_crop:
        start = self._rnd.randint(0, fcount - T)
      else:
        start = 0
    else:
      start = 0

    # Uninteresting episodes.
    if fcount < T // 2:
      return None

    if np.max(y_s) < 5:
      return None

    # Randomly flip the sequence.
    if self._random_flip:
      reverse = self._rnd.randint(0, 2) == 1
    else:
      reverse = False

    def slice(x, start, end, reverse):
      x_ = x[start:end]
      if reverse:
        x_ = x_[::-1]
      assert x.shape[0] > 0
      return x_

    x_s = slice(x_s, start, start + T, reverse)
    x_att = slice(x_att, start, start + T, reverse)
    y_s = slice(y_s, start, start + T, reverse)
    bboxes = slice(bboxes, start, start + T, reverse)
    categories = slice(categories, start, start + T, reverse)
    instance_ids = slice(instance_ids, start, start + T, reverse)
    flag = slice(flag, start, start + T, reverse)

    def query_np(x, lbl_map, unk_id):
      x = np.expand_dims(x, 1)  # [T, 1]
      x_eq = np.equal(x, lbl_map).astype(np.float32)  # [T, N]
      x_valid = np.sum(x_eq, axis=1)  # [T]

      # Everything that has not been found -> fixed unknown.
      # This means it's a distractor.
      x = np.argmax(x_eq, axis=1).astype(np.float32)
      x = x_valid * x + (1 - x_valid) * unk_id
      x = x.astype(np.int32)
      return x

    # Semisupervised episode.
    if self._semisupervised:
      labeled = (self.sample_label_mask(y_s,
                                        self._label_ratio)).astype(np.bool)
      unlabeled = np.logical_not(labeled)

      label_map, y_s_label = tf.unique(y_s[labeled])
      label_map = label_map.numpy()

      y_s[labeled] = y_s_label
      if len(y_s[unlabeled]) > 0:
        y_s[unlabeled] = query_np(y_s[unlabeled], label_map, unk_id)
      y_s_orig = np.copy(y_s)
      y_s2 = np.copy(y_s)

      y_s[unlabeled] = unk_id
      y_s2[unlabeled] = -1
    else:
      # Recompute y_s
      label_map, y_s = tf.unique(y_s)
      y_s = y_s.numpy()
      y_s2 = np.copy(y_s)
      y_s_orig = np.copy(y_s)

    # Compute Groundtruth.
    y_gt = np.zeros([len(y_s)], dtype=np.int32)
    cummax = np.maximum.accumulate(y_s2)
    y_gt[0] = unk_id
    cond = y_s_orig[1:] > cummax[:-1]
    y_gt[1:] = np.where(cond, unk_id, y_s_orig[1:])

    # Compute y_gt.
    y_full = y_s_orig
    episode = {
        'x_s': x_s,  # RGB image, mask as the 4th channel.
        'x_att': x_att,  # Attention mask.
        'y_s': y_s,
        'y_gt': y_gt,
        'y_full': y_full,
        'bbox': bboxes,
        'category': categories,
        'instance_id': instance_ids,
        'flag_s': flag
    }
    # [print(z) for z in list(zip(range(100), y_full, instance_ids, flag))]
    return episode

  def get_labels(self, episode):
    pass

  def _next(self):
    """Next example."""
    # Get one ID.
    item = None
    while item is None:
      idx = self.sampler.sample_collection(self.batch_size)
      if idx is None:
        break
      else:
        idx = idx[0]
      item = self.process_one(idx)
    return item

  def __iter__(self):
    return self._tf_dataset_iter

  def __len__(self):
    return self.dataset.get_size()

  def get_generator(self):
    """Gets generator function, for tensorflow Dataset object."""
    while True:
      item = self._next()
      if item is not None:
        yield item
      else:
        break

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dummy = self._next()
    self.sampler.reset()
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict([(k, tf.shape(dummy[k])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)

    def preprocess(data):
      # Combine preprocessed RGB image and attention map (in float format).
      data['x_s'] = tf.concat(
          [self.preprocessor(data['x_s']),
           tf.cast(data['x_att'], tf.float32)],
          axis=-1)
      del data['x_att']
      return data

    if self.preprocessor is not None:
      ds = ds.map(preprocess)
    ds = ds.batch(self.batch_size)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  def reset(self):
    """Resets sampler."""
    self.sampler.reset()

  @property
  def kwargs(self):
    """Additional parameters for the sampler."""
    return self._kwargs

  @property
  def preprocessor(self):
    """Image preprocessor."""
    return self._preprocessor

  @property
  def sampler(self):
    """Episode sampler."""
    return self._sampler

  @property
  def dataset(self):
    """Dataset source."""
    return self._dataset

  @property
  def nclasses(self):
    """Number of classes per episode."""
    return self._nclasses

  @property
  def batch_size(self):
    """Number of episodes."""
    return self._batch_size

  @property
  def maxlen(self):
    """Max length of the sequence."""
    return self._maxlen

  @property
  def tf_dataset(self):
    return self._tf_dataset


if __name__ == '__main__':
  from fewshot.data.datasets.matterport import MatterportDataset
  from fewshot.data.samplers.minibatch_sampler import MinibatchSampler
  from fewshot.data.preprocessors.normalization_preprocessor import NormalizationPreprocessor  # NOQA
  sampler = MinibatchSampler(0)
  dataset = MatterportDataset("./data/matterport3d/fewshot/h5_data", "val")
  it = SimEpisodeIterator(
      dataset,
      sampler,
      2,
      30,
      preprocessor=NormalizationPreprocessor(),
      fix_unknown=True,
      maxlen=100,
      semisupervised=False,
      prefetch=True,
      random_crop=True)
  for i, d in zip(range(2), it):
    pass
