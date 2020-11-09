"""Visualize an online few-shot episode.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage example:
  python -m fewshot.viz.plot_episode --env ./config/omniglot.prototxt \
      --data ./config/data/omniglot-crp.prototxt \
      --output ./output/ssl_ \
      --restore ./results/protonet
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches as patches

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.utils.logger import get as get_logger

log = get_logger()


def plot_one_episode(episode,
                     fname,
                     model=None,
                     ncol=8,
                     nrow=5,
                     nway=10,
                     nstage=1):
  """Plots a single episode.
  Unlabeled examples will have red bounding box.
  Also assuming that there is no query set.

  Args:
    episode: A few-shot episode.
    fname: String. File name to save the picture.
  """
  img = episode['x_s'].numpy()
  print(img.shape, img.dtype)
  lbl = episode['y_s'].numpy()
  lbl_gt = episode['y_gt'].numpy()
  print()

  if 'y_dis' in episode:
    distractor = episode['y_dis'].numpy()
  else:
    distractor = None

  flag = episode['flag_s'].numpy()

  # Hierarchical stage information.
  print(episode.keys())
  if 'stage_id' in episode:
    is_hierarchical = True
    stage_id = episode['stage_id'].numpy()
    print(stage_id)
    class_id_in_stage = episode['in_stage_class_id'].numpy()
  else:
    is_hierarchical = False

  fig = plt.figure(figsize=(int(ncol * 1.5), int(nrow * 1.5)))
  assert len(img.shape) == 5, 'Expect train_images rank = 5'

  if model is not None:
    pred = model.eval_step(episode['x_s'], episode['y_s'], None)
    pred = pred.numpy()
    # print('pred', pred.shape)
  else:
    pred = None

  print(flag[0])
  # assert False
  for i in range(1, 1 + min(ncol * nrow, img.shape[1])):

    if img.shape[-1] == 1:
      img_i = img[0, i - 1, :, :, 0]
      img_i = 0.5 * (img_i + 1.0)
    elif img.shape[-1] == 4:
      img_i = img[0, i - 1, :, :, :]
      img_i[:, :, :3] = 0.5 * (img_i[:, :, :3] + 1.0)
      img_i[:, :, -1] = 0.5 * (img_i[:, :, -1] + 1.0)
    lbl_i = lbl[0, i - 1]
    lbl_gt_i = lbl_gt[0, i - 1]
    flag_i = flag[0, i - 1]

    if distractor is not None:
      distractor_i = distractor[0, i - 1]

    if pred is not None:
      pred_i = np.argmax(pred[0, i - 1], axis=-1)
    if flag_i == 0:  # End of the episode.
      # print('end', i)
      break
    ax = plt.subplot(nrow, ncol, i)

    if img.shape[-1] == 1:
      ax.imshow(img_i, cmap='Greys')
    else:
      ax.imshow(img_i)

    edgecolor_list = np.array([[141, 153, 174], [217, 4, 41]]) / 255.0
    edgecolor = edgecolor_list[1]

    if distractor is not None and distractor_i == 1:
      edgecolor = 'gray'  # Distractor
    elif lbl_i == nway:
      edgecolor = edgecolor_list[0]  # Unlabeled
    # edgecolor = 'gray'  # Distractor

    rect = patches.Rectangle(
        (0, 0),
        img_i.shape[1] - 1,
        img_i.shape[0] - 1,
        linewidth=max(int(img.shape[2] / 30), 2),
        edgecolor=edgecolor,
        # linestyle='-',
        linestyle=':' if lbl_i == nway else '-',
        facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.axis('off')
    # title = 't={}\\y={}'.format(i - 1,
    #                             str(lbl_gt_i) if lbl_gt_i != nway else 'unk')
    title = '{}'.format(i - 1)
    # if pred is None:
    #   # print(i - 1, lbl_i)
    #   title = 't={}\\y={}/{}'.format(i - 1, lbl_i, lbl_gt_i)
    # else:
    #   title = 't={}\\y={}/{} p={}'.format(i - 1, lbl_i, lbl_gt_i, pred_i)
    # Indicate correct/wrong.
    # if pred_i == lbl_gt_i:
    #   edgecolor = 'green'
    # else:
    #   edgecolor = 'red'
    # print(edgecolor)
    # rect = patches.Rectangle((0, img_i.shape[0] - 4),
    #                          3,
    #                          3,
    #                          facecolor=edgecolor)
    # ax.add_patch(rect)
    ax.set_title(title)

    # cmap = np.array([[61, 220, 151], [70, 35, 122], [37, 110, 255],
    #                  [252, 252, 252], [255, 73, 92]]) / 255.0
    cmap = np.array([[61, 220, 151], [70, 35, 122], [37, 110, 255],
                     [216, 216, 244], [139, 148, 163]]) / 255.0
    stagename = ['A', 'B', 'C', 'D', 'E']
    stage_text = ['black', 'white', 'white', 'black', 'white']

    # Color box for stage.
    if is_hierarchical:
      stage_i = stage_id[0, i - 1]
      class_i = class_id_in_stage[0, i - 1]
      stage_f = float(stage_i) / stage_id.max()
      # print(stage_i, class_i)
      # stage_color = plt.get_cmap('gist_rainbow')(stage_f)
      stage_color = cmap[stage_i]
      rect = patches.Rectangle(
          (-1, -1),
          12,
          11,
          facecolor=stage_color,
      )
      ax.add_patch(rect)
      # ax.text(2, 5, str(class_i), color='white', fontsize=12)
      ax.text(
          1.5,
          6,
          stagename[stage_i] + str(class_i),
          color=stage_text[stage_i],
          fontsize=15)
    # plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

  for ext in ['.pdf', '.png']:
    plt.savefig(fname + ext)
    print('Saved to {}{}'.format(fname, ext))


def plot_episodes(data_it, nepisode, fname_tmpl, **kwargs):
  """Plots the episode.

  Args:
    data_it: Episode iterators.
    nepisode: Int. Number of episodes to plot.
    fname_fmpl: String. Output file name template.
  """
  for i, e in zip(range(nepisode), data_it):
    plot_one_episode(e, fname_tmpl.format(i), **kwargs)


def main():
  data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)

  # Get dataset.
  dataset = get_data_fs(env_config, load_train=True)

  # Get data iterators.
  if data_config.hierarchical:
    save_additional_info = True
  else:
    save_additional_info = False

  if env_config.dataset in ["matterport"]:
    data = get_dataiter_sim(
        dataset,
        data_config,
        batch_size=1,
        nchw=False,
        distributed=False,
        prefetch=False)
  else:
    data = get_dataiter_continual(
        dataset,
        data_config,
        batch_size=1,
        nchw=False,
        prefetch=False,
        save_additional_info=save_additional_info,
        random_box=data_config.random_box)

  if not os.path.exists(args.output):
    os.makedirs(args.output)
    os.makedirs(os.path.join(args.output, 'val'))
    os.makedirs(os.path.join(args.output, 'train'))

  if args.restore is None:
    log.info('No model provided.')
    fs_model = None
  else:
    config = os.path.join(args.restore, 'config.prototxt')
    config = get_config(config, ExperimentConfig)
    model = build_pretrain_net(config)
    fs_model = build_net(config, backbone=model.backbone)
    fs_model.load(latest_file(args.restore, 'weights-'))

  if data_config.maxlen <= 40:
    ncol = 5
    nrow = 8
  elif data_config.maxlen <= 100:
    ncol = 6
    nrow = 10
  else:
    ncol = 6
    nrow = 25

  # Plot episodes.
  plot_episodes(
      data['val_fs'],
      args.nepisode,
      os.path.join(args.output, 'val', 'episode_{:06d}'),
      model=fs_model,
      nway=data_config.nway,
      ncol=ncol,
      nrow=nrow,
      nstage=data_config.nstage)

  # Plot episodes.
  plot_episodes(
      data['train_fs'],
      args.nepisode,
      os.path.join(args.output, 'train', 'episode_{:06d}'),
      model=fs_model,
      nway=data_config.nway,
      ncol=ncol,
      nrow=nrow,
      nstage=data_config.nstage)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Lifelong Few-Shot Training')
  # Whether allow images to repeat in an episode.
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--nepisode', type=int, default=10)
  parser.add_argument('--output', type=str, default='./output')
  parser.add_argument('--restore', type=str, default=None)
  args = parser.parse_args()
  main()
