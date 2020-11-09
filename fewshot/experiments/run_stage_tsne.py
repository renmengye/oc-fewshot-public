"""
Run offline logistic regression or MLP at each timestep, as an oracle.
This works like a performance upper bound.
Also this is closed set setting. Assuming fully labeled.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import numpy as np
import os
import pickle as pkl
import six
import tensorflow as tf
from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.utils.logger import get as get_logger
from fewshot.models.model_factory import get_module
from fewshot.models.modules.rnn_encoder import RNNEncoder
from fewshot.models.nets.rnn_encoder_net import RNNEncoderNet

log = get_logger()


def evaluate(f, K, dataiter, num_steps):
  """Evaluates online few-shot episodes.

  Args:
    model: Model instance.
    dataiter: Dataset iterator.
    num_steps: Number of episodes.
  """
  if num_steps == -1:
    it = six.moves.xrange(len(dataiter))
  else:
    it = six.moves.xrange(num_steps)
  it = tqdm(it, ncols=0)
  results = []
  for i, batch in zip(it, dataiter):
    # Get features.
    h = f(batch['x_s'], batch['y_s'])
    if type(h) is tuple:
      h, (beta, gamma, beta2, gamma2, count) = h
      print('beta/count', np.stack([beta, count], axis=-1))
      batch['beta'] = beta.numpy()
      batch['gamma'] = gamma.numpy()
      batch['beta2'] = beta2.numpy()
      batch['gamma2'] = gamma2.numpy()
      batch['count'] = count.numpy()
    batch['h'] = h.numpy()
    results.append(batch)
  return results


def visualize(data_list, output_folder):
  import sklearn.manifold
  from matplotlib import pyplot as plt

  M = 1
  has_stage = 'stage_id_gt' in data_list['test_fs'][0]
  for nplot in range(10):
    Xall = []
    yall = []
    ysall = []
    sall = []
    call = []
    for i in range(nplot * M, (nplot + 1) * M):
      data = data_list['test_fs'][i]
      flag = data['flag_s'].numpy()
      T = flag.sum()
      X = data['h'][0, :T]
      y = data['y_full'].numpy()[0, :T]
      ys = data['y_s'].numpy()[0, :T]
      Xall.append(X)
      yall.append(y + (i - nplot * M) * 40)
      ysall.append(ys)

      if has_stage:
        sall.append(data['stage_id'].numpy()[0, :T])
        call.append(data['in_stage_class_id'].numpy()[0, :T])

    if M > 1:
      X = np.concatenate(Xall, axis=0)
      y = np.concatenate(yall, axis=0)
      ys = np.concatenate(ysall, axis=0)

      if has_stage:
        s = np.concatenate(sall, axis=0)
        c = np.concatenate(call, axis=0)
    else:
      X = Xall[0]
      y = yall[0]
      ys = ysall[0]
      if has_stage:
        s = np.concatenate(sall, axis=0)
        c = np.concatenate(call, axis=0)

    if has_stage:
      color = s * 10 + c
    else:
      color = y

    maxidx = data['y_gt'].numpy().max()
    labeled = ys < maxidx
    unlabeled = ys == maxidx
    tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, random_state=1234)
    pca = sklearn.decomposition.PCA(n_components=2)
    Z = tsne.fit_transform(X)
    fig = plt.figure(figsize=(8, 5))
    cmap = 'rainbow'
    plt.scatter(
        Z[labeled][:, 0],
        Z[labeled][:, 1],
        c=color[labeled],
        cmap=cmap,
        alpha=1.0,
        s=50,
        linewidths=2)
    plt.scatter(
        Z[unlabeled][:, 0],
        Z[unlabeled][:, 1],
        c=color[unlabeled],
        cmap=cmap,
        alpha=0.3,
        s=15,
        linewidths=2)
    plt.xticks([])
    plt.yticks([])

    for j in range(X.shape[0]):
      if 'stage_id_gt' in data_list['test_fs'][0]:
        cls_txt = '{:d}'.format(
            data_list['test_fs'][nplot]['in_stage_class_id'][0, j])
        plt.annotate(cls_txt, (Z[j, 0], Z[j, 1]), fontsize=8)
      else:
        plt.annotate('{:d}'.format(y[j]), (Z[j, 0], Z[j, 1]), fontsize=8)
    plt.savefig(os.path.join(output_folder, 'tsne-{:03d}.pdf'.format(nplot)))


def main():
  assert tf.executing_eagerly(), 'Only eager mode is supported.'
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.data is not None, 'You need to pass in episode config file path'
  assert args.env is not None, 'You need to pass in environ config file path'
  assert args.tag is not None, 'You need to specify a tag'

  log.info('Command line args {}'.format(args))
  config = get_config(args.config, ExperimentConfig)
  data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  log.info('Model: \n{}'.format(config))
  log.info('Data episode: \n{}'.format(data_config))
  log.info('Environment: \n{}'.format(env_config))
  config.num_classes = data_config.nway  # Assign num classes.
  config.num_steps = data_config.maxlen
  config.memory_net_config.max_classes = data_config.nway
  config.memory_net_config.max_stages = data_config.nstage
  config.memory_net_config.max_items = data_config.maxlen
  config.oml_config.num_classes = data_config.nway
  config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.
  log.info('Number of classes {}'.format(data_config.nway))
  if 'SLURM_JOB_ID' in os.environ:
    log.info('SLURM job ID: {}'.format(os.environ['SLURM_JOB_ID']))

  # Create save folder.
  save_folder = os.path.join(env_config.results, env_config.dataset, args.tag)
  results_file = os.path.join(save_folder, 'results_tsne.pkl')

  # Plot features
  if os.path.exists(results_file):
    batch_list = pkl.load(open(results_file, 'rb'))
    return visualize(batch_list, save_folder)

  # To get CNN features from.
  model = build_pretrain_net(config)
  if args.rnn:
    rnn_memory = get_module(
        "lstm",
        "lstm",
        model.backbone.get_output_dimension()[0],
        config.lstm_config.hidden_dim,
        layernorm=config.lstm_config.layernorm,
        dtype=tf.float32)
    proto_memory = get_module(
        'ssl_min_dist_proto_memory',
        'proto_memory',
        config.memory_net_config.radius_init,
        max_classes=config.memory_net_config.max_classes,
        fix_unknown=config.fix_unknown,
        unknown_id=config.num_classes if config.fix_unknown else None,
        similarity=config.memory_net_config.similarity,
        static_beta_gamma=not config.hybrid_config.use_pred_beta_gamma,
        dtype=tf.float32)
    memory = RNNEncoder(
        "proto_plus_rnn_ssl_v4",
        rnn_memory,
        proto_memory,
        readout_type=config.mann_config.readout_type,
        use_pred_beta_gamma=config.hybrid_config.use_pred_beta_gamma,
        use_feature_fuse=config.hybrid_config.use_feature_fuse,
        use_feature_fuse_gate=config.hybrid_config.use_feature_fuse_gate,
        use_feature_scaling=config.hybrid_config.use_feature_scaling,
        use_feature_memory_only=config.hybrid_config.use_feature_memory_only,
        skip_unk_memory_update=config.hybrid_config.skip_unk_memory_update,
        use_ssl=config.hybrid_config.use_ssl,
        use_ssl_beta_gamma_write=config.hybrid_config.use_ssl_beta_gamma_write,
        use_ssl_temp=config.hybrid_config.use_ssl_temp,
        dtype=tf.float32)
    rnn_model = RNNEncoderNet(config, model.backbone, memory)
    f = lambda x, y: rnn_model.forward(x, y, is_training=False)  # NOQA
  else:
    f = lambda x, y: model.backbone(x[0], is_training=False)[None, :, :] # NOQA

  K = config.num_classes
  # Get dataset.
  dataset = get_data_fs(env_config, load_train=True)

  # Get data iterators.
  if env_config.dataset in ["matterport"]:
    data = get_dataiter_sim(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=model.backbone.config.data_format == 'NCHW',
        seed=args.seed)
  else:
    data = get_dataiter_continual(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=model.backbone.config.data_format == 'NCHW',
        save_additional_info=True,
        random_box=data_config.random_box,
        seed=args.seed)

  if args.usebest:
    latest = latest_file(save_folder, 'best-')
  else:
    latest = latest_file(save_folder, 'weights-')

  if args.rnn:
    rnn_model.load(latest)
  else:
    model.load(latest)

  data['trainval_fs'].reset()
  data['val_fs'].reset()
  data['test_fs'].reset()

  results_all = {}
  split_list = ['test_fs']
  name_list = ['Test']
  nepisode_list = [100]

  for split, name, N in zip(split_list, name_list, nepisode_list):
    data[split].reset()

    data_fname = '{}/{}.pkl'.format(env_config.data_folder, split)
    if not os.path.exists(data_fname):
      batch_list = []
      for i, batch in zip(range(N), data[split]):
        for k in batch.keys():
          batch[k] = batch[k]
        batch_list.append(batch)
      pkl.dump(batch_list, open(data_fname, 'wb'))

    batch_list = pkl.load(open(data_fname, 'rb'))
    r1 = evaluate(f, K, batch_list, N)
    results_all[split] = r1
  pkl.dump(results_all, open(results_file, 'wb'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Lifelong Few-Shot Training')
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--reeval', action='store_true')
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--testonly', action='store_true')
  parser.add_argument('--rnn', action='store_true')
  parser.add_argument('--usebest', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  tf.random.set_seed(1234)
  main()
