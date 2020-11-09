"""Matterport3D dataset API.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import glob
import h5py
import json
import numpy as np
import os

from fewshot.data.registry import RegisterDataset


@RegisterDataset("roaming-rooms")
@RegisterDataset("matterport")  # Legacy name
class MatterportDataset(object):

  def __init__(self, folder, split, dirpath="fewshot/data/matterport_split"):
    assert folder is not None
    assert split is not None
    self._folder = folder
    self._split = split
    assert split in ["train", "val", "test"]
    split_file = os.path.join(dirpath, split + '.txt')
    with open(split_file, "r") as f:
      envs = f.readlines()
    envs = set(map(lambda x: x.strip("\n"), envs))
    all_h5_files = glob.glob(os.path.join(folder, "*", "*__imgs.h5"))
    files_in_split = sorted(
        filter(lambda x: x.split("_")[-8].split(".")[0] in envs, all_h5_files))

    print(split, folder, len(files_in_split))
    assert len(files_in_split) > 0

    def make_json(x):
      return x.replace('__imgs.h5', '__annotations.json')

    basename_list = [
        ele.split("__imgs.")[0]
        for ele in files_in_split
        if os.path.exists(make_json(ele))
    ]
    self._file_list = basename_list
    # print(split, len(self._file_list))

  def _make_iter(self, img_arr, l_arr):
    """Makes an PNG encoding string iterator."""
    prev = 0
    l_cum = np.cumsum(l_arr)
    for i, idx in enumerate(l_cum):
      yield cv2.imdecode(img_arr[prev:idx], -1)
      prev = idx

  def get_episode(self, idx):
    """Get a single episode file."""
    basename = self.file_list[idx]
    h5_fname = basename + "__imgs.h5"
    json_fname = basename + "__annotations.json"
    with h5py.File(h5_fname, "r") as f, open(json_fname, "r") as f2:
      jsond = json.load(f2)
      inst_seg = f["instance_segmentation"][:]
      inst_seg_len = f["instance_segmentation_len"][:]
      rgb = f["matterport_RGB"][:]
      rgb_len = f["matterport_RGB_len"][:]

    rgb_iter = self._make_iter(rgb, rgb_len)
    inst_seg_iter = self._make_iter(inst_seg, inst_seg_len)
    data = []
    iter_ = enumerate(zip(rgb_iter, inst_seg_iter, jsond))
    for i, (rgb_, inst_seg_, annotation_) in iter_:
      data.append({
          "instance_seg": inst_seg_,
          "rgb": rgb_,
          "annotation": annotation_
      })
    return data

  def get_size(self):
    """Gets the total number of images."""
    return len(self.file_list)

  @property
  def folder(self):
    """Data folder."""
    return self._folder

  @property
  def split(self):
    """Data split."""
    return self._split

  @property
  def file_list(self):
    return self._file_list


if __name__ == "__main__":
  from tqdm import tqdm
  sss_total = 0
  for sp in ['train', 'val', 'test']:
    sss = 0
    dataset = MatterportDataset("./data/matterport3d/fewshot/h5_data", sp)
    for i in tqdm(range(dataset.get_size())):
      sss += dataset.get_episode(i)
    sss_total += sss
