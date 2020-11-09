"""
Run offline logistic regression at each timestep, as an oracle.
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
import sklearn.linear_model
from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.get_stats import get_stats
from fewshot.experiments.get_stats import log_results
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.utils.logger import get as get_logger
from fewshot.models.model_factory import get_module
from fewshot.models.modules.rnn_encoder import RNNEncoder
from fewshot.models.nets.rnn_encoder_net import RNNEncoderNet

log = get_logger()


def eval_step(x, y, ys, K, T):
  """Leave one out at each step."""
  assert x.shape[0] == 1
  y = y
  ys = ys
  T2 = x.shape[1]
  x = x[:, :T]
  y = y[:, :T]
  ys = ys[:, :T]
  ypred = np.zeros([1, T2, K + 1], dtype=np.float32)
  ypred_id = np.zeros([1, T2], dtype=np.int64)
  ys[ys == K] = -1

  def loo(data, t):
    data = np.expand_dims(data[ys != -1], 0)
    if t < data.shape[1] - 1:
      return np.concatenate([data[0, :t], data[0, t + 1:T]])
    else:
      return data[0, :t]

  def mask(y, k, K):
    yexp = np.exp(y)
    ymask = np.zeros([y.shape[0], K + 1]) - 1e5
    if k < y.shape[1]:
      ymask[:, :k] = y[:, :k]
      ymask[:, K] = np.log(1.0 - yexp[:, :k].max() + 1e-6)
    else:
      ymask[:, :y.shape[1]] = y
    renom = np.exp(ymask)
    renom = (renom + 1e-6) / np.sum(renom)
    return renom

  for t in range(T):
    x_ = loo(x, t)
    y_ = loo(y, t)
    if t == 0 or np.max(ys[0, :t]) == -1:
      __ = np.zeros([1, K + 1]) - 1e5
      __[0, K] = 0.0
      ypred[0, t] = __
      ypred_id[0, t] = K
      continue

    k = np.max(ys[0, :t]) + 1
    sel = y_ <= np.maximum(k, 5)
    x_ = x_[sel]
    y_ = y_[sel]
    lr = sklearn.linear_model.LogisticRegression(max_iter=10000)
    lr.fit(x_, y_)
    xtest_ = x[:, t]
    ypred_ = lr.predict_log_proba(xtest_)
    ypred_ = mask(ypred_, k, K)
    ypred[0, t] = ypred_

    if ypred_[0, -1] > 0.5:
      ypred_id_ = K
    else:
      ypred_id_ = np.argmax(ypred_[0, :-1])
    ypred_id[0, t] = ypred_id_

  return ypred, ypred_id


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
    x = batch['x_s']
    y_full = batch['y_full'].numpy()
    y = batch['y_s'].numpy()
    y_gt = batch['y_gt'].numpy()
    flag = batch['flag_s'].numpy()

    # Get features.
    x = f(x).numpy()
    T = flag.sum()
    pred, pred_id = eval_step(x, y_full, y, K, T)

    results.append({
        'y_full': y_full,
        'y_gt': y_gt,
        'y_s': y,
        'pred': pred,
        'pred_id': pred_id,
        'flag': flag
    })
  return results


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
  results_file = os.path.join(save_folder, 'results.pkl')
  logfile = os.path.join(save_folder, 'results.csv')

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
    memory = RNNEncoder(
        "proto_plus_rnn_ssl_v4",
        rnn_memory,
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
    f = lambda x: rnn_model.forward(x, is_training=False)  # NOQA
  else:
    f = lambda x: model.backbone(x[0], is_training=False)[None, :, :]  # NOQA

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
  if os.path.exists(results_file) and args.reeval:
    # Re-display results.
    results_all = pkl.load(open(results_file, 'rb'))
    for split, name in zip(['trainval_fs', 'val_fs', 'test_fs'],
                           ['Train', 'Val', 'Test']):
      stats = get_stats(results_all[split], tmax=data_config.maxlen)
      log_results(stats, prefix=name, filename=logfile)
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
    if args.testonly:
      split_list = ['test_fs']
      name_list = ['Test']
      nepisode_list = [config.num_episodes]
    else:
      split_list = ['trainval_fs', 'val_fs', 'test_fs']
      name_list = ['Train', 'Val', 'Test']
      nepisode_list = [600, config.num_episodes, config.num_episodes]

    for split, name, N in zip(split_list, name_list, nepisode_list):
      data[split].reset()
      r1 = evaluate(f, K, data[split], N)
      stats = get_stats(r1, tmax=data_config.maxlen)
      log_results(stats, prefix=name, filename=logfile)
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
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  tf.random.set_seed(1234)
  main()
