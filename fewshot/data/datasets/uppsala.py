"""Uppsala texture dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import os
import numpy as np
from tqdm import tqdm
from fewshot.data.datasets.pickle_cache_dataset import PickleCacheDataset
from fewshot.data.registry import RegisterDataset


@RegisterDataset('uppsala')
class UppsalaDataset(PickleCacheDataset):

  def __init__(self, folder, split):
    super(UppsalaDataset, self).__init__(folder, split)
    self._cls_dict = self.get_cls_dict()

  def _read_dataset(self):
    folder, split = self._folder, self._split
    curdir = os.path.dirname(os.path.realpath(__file__))
    split_file = os.path.join(curdir, '../uppsala_split',
                              '{}.txt'.format(split))
    lines = open(split_file, 'r').readlines()
    lines = map(lambda x: x.strip('\n\r'), lines)
    data = []
    label_idx = []
    label_str = []
    for ii, ll in enumerate(tqdm(lines, ncols=0, desc="Reading data")):
      subfolder = os.path.join(folder, ll)
      img_list = os.listdir(subfolder)
      for img_fname in img_list:
        fname_ = os.path.join(subfolder, img_fname)
        img = cv2.imread(fname_)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = np.minimum(255, np.maximum(0, img))
        data.append(img)
        label_idx.append(ii)
        label_str.append(ll)
    return {
        'images': np.stack(data, axis=0),
        'labels': np.array(label_idx, dtype=np.int32),
        'label_str': label_str
    }

  def get_images(self, inds):
    """Gets an image or a batch of images."""
    return self.images[inds]

  def get_cls_dict(self):
    """Computes class dictionary."""
    cls_dict = {}
    for i, l in enumerate(self.labels):
      if l in cls_dict:
        cls_dict[l].append(i)
      else:
        cls_dict[l] = [i]
    return cls_dict

  @property
  def cls_dict(self):
    return self._cls_dict


if __name__ == "__main__":
  for split in ["train", "val", "test"]:
    UppsalaDataset("./data/uppsala-texture", split)
