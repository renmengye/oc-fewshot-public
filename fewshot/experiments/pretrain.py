"""
Pretrain a network on regular classification.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
  pretrain.py --config [CONFIG] --tag [TAG} --dataset [DATASET] \
              --data_folder [DATA FOLDER] --results [SAVE FOLDER]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import numpy as np
import six
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.build_model import build_fewshot_net
from fewshot.experiments.get_data_iter import get_dataiter_fewshot
from fewshot.experiments.get_data_iter import get_dataiter
from fewshot.experiments.utils import ExperimentLogger
from fewshot.experiments.utils import save_config
from fewshot.experiments.utils import latest_file
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data
from fewshot.experiments.utils import get_data_fs
from fewshot.utils.logger import get as get_logger

log = get_logger()


def label_equal(pred, label, axis=-1):
  return pred == label.astype(pred.dtype)


def top1_correct(pred, label, axis=-1):
  """Calculates top 1 correctness."""
  assert pred.shape[0] == label.shape[0], '{} != {}'.format(
      pred.shape[0], label.shape[0])
  pred_idx = np.argmax(pred, axis=axis)
  return pred_idx == label.astype(pred_idx.dtype)


def top1_acc(pred, label, axis=-1):
  """Calculates top 1 accuracy."""
  return top1_correct(pred, label, axis=axis).mean()


def topk_acc(pred, label, k, axis=-1):
  """Calculates top 5 accuracy."""
  assert pred.shape[0] == label.shape[0], '{} != {}'.format(
      pred.shape[0], label.shape[0])
  topk_choices = np.argsort(pred, axis=axis)
  if len(topk_choices.shape) == 2:
    topk_choices = topk_choices[:, ::-1][:, :k]
  elif len(topk_choices.shape) == 3:
    topk_choices = topk_choices[:, :, ::-1][:, :, :k]
  else:
    raise NotImplementedError()
  return np.sum(topk_choices == np.expand_dims(label, axis), axis=axis).mean()


def stderr(array, axis=0):
  """Calculates standard error."""
  return array.std(axis=axis) / np.sqrt(float(array.shape[0]))


def evaluate(model, dataiter, num_steps, verbose=False):
  """Evaluates accuracy."""
  acc_list = []
  acc_top5_list = []
  batch_size = model.config.optimizer_config.batch_size
  if num_steps <= 0:
    # Determine the total number of steps manually.
    num_steps = len(dataiter)
  if verbose:
    it = tqdm(range(num_steps), ncols=0)
  else:
    it = range(num_steps)
  for ii, batch in zip(it, dataiter):
    x = batch['x']
    y = batch['y']
    prediction_a = model.eval_step(x).numpy()
    acc_list.append(top1_acc(prediction_a, y.numpy()))
    acc_top5_list.append(topk_acc(prediction_a, y.numpy(), 5))
    if verbose:
      it.set_postfix(acc=u'{:.3f}±{:.3f}'.format(
          np.array(acc_list).mean() * 100.0,
          stderr(np.array(acc_list)) * 100.0))
  acc_list = np.array(acc_list)
  acc_top5_list = np.array(acc_top5_list)
  results_dict = {
      'acc': acc_list.mean(),
      'acc_se': stderr(acc_list),
      'acc_top5': acc_top5_list.mean(),
      'acc_top5_se': stderr(acc_top5_list)
  }
  return results_dict


def evaluate_fs(model, dataiter, num_steps, verbose=False):
  """Evaluates few-shot."""
  acc_list = np.zeros([num_steps])
  acc_top5_list = np.zeros([num_steps])
  if verbose:
    it = tqdm(six.moves.xrange(num_steps), ncols=0)
  else:
    it = six.moves.xrange(num_steps)
  for tt, batch in zip(it, dataiter):
    x = batch['x_s']
    y = batch['y_s']
    x_test = batch['x_q']
    y_test = batch['y_q']
    prediction_a = model.eval_step(x, y, x_test)
    # print(model, prediction_a)
    if type(prediction_a) == tuple:
      pred_train = prediction_a[0].numpy()
      prediction_a = prediction_a[1].numpy()
    else:
      pred_train = None
      prediction_a = prediction_a.numpy()

    acc_list[tt] = top1_acc(prediction_a, y_test[0].numpy())
    if pred_train is not None:
      acc_train = top1_acc(pred_train, y.numpy())
    else:
      acc_train = 0.0
    # acc_top5_list[tt] = topk_acc(prediction_a, y_test.numpy(), 5)
    if verbose:
      it.set_postfix(
          acc_train='{:.3f}'.format(acc_train * 100.0),
          acc=u'{:.3f}±{:.3f}'.format(acc_list[:tt + 1].mean() * 100.0,
                                      stderr(acc_list[:tt + 1]) * 100.0))
  acc_list = acc_list[:tt]
  # acc_top5_list = acc_top5_list[:tt]
  results_dict = {
      'acc': acc_list.mean(),
      'acc_se': stderr(acc_list),
      # 'acc_top5': acc_top5_list.mean(),
      # 'acc_top5_se': stderr(acc_top5_list)
  }
  return results_dict


def train(model,
          dataiter,
          dataiter_test,
          save_folder,
          proto_net=None,
          dataiter_fs=None,
          dataiter_test_fs=None,
          reload_flag=None):
  """Trains the model."""
  N = model.config.optimizer_config.max_train_steps
  config = model.config.train_config
  start = model.step.numpy()
  if start > 0:
    log.info('Restore from step {}'.format(start))
  it = six.moves.xrange(start, N)
  writer = tf.summary.create_file_writer(save_folder)
  logger = ExperimentLogger(writer)
  rtrain = None

  with writer.as_default():
    it = tqdm(it, ncols=0)
    for ii, batch in zip(it, dataiter):
      x = batch['x']
      y = batch['y']
      loss = model.train_step(x, y)

      # Reload model weights.
      if ii == start and reload_flag is not None:
        model.load(reload_flag, load_optimizer=True)

      # Evaluate.
      if (ii + 1) % config.steps_per_val == 0 or ii == 0:
        rtrain = evaluate(model, dataiter, 100)
        logger.log('acc train', ii + 1, rtrain['acc'] * 100.0)
        logger.log('lr', ii + 1, model.learn_rate())

        if dataiter_test is not None:
          dataiter_test.reset()
          rtest = evaluate(model, dataiter_test, -1)
          logger.log('acc val', ii + 1, rtest['acc'] * 100.0)

        if proto_net is not None and dataiter_fs is not None:
          rtrain_fs = evaluate_fs(proto_net, dataiter_fs, 120)
          rtest_fs = evaluate_fs(proto_net, dataiter_test_fs, 120)
          logger.log('fs acc train', ii + 1, rtrain_fs['acc'] * 100.0)
          logger.log('fs acc val', ii + 1, rtest_fs['acc'] * 100.0)
        print()

      # Save.
      if (ii + 1) % config.steps_per_save == 0 or ii == 0:
        model.save(os.path.join(save_folder, 'weights-{}'.format(ii + 1)))

      # Write logs.
      if (ii + 1) % config.steps_per_log == 0 or ii == 0:
        logger.log('loss', ii + 1, loss)
        logger.flush()

        # Update progress bar.
        post_fix_dict = {}
        if rtrain is not None:
          post_fix_dict['acc_t'] = '{:.1f}'.format(rtrain['acc'] * 100.0)
          if dataiter_test is not None:
            post_fix_dict['acc_v'] = '{:.1f}'.format(rtest['acc'] * 100.0)
        post_fix_dict['lr'] = '{:.1e}'.format(model.learn_rate())
        post_fix_dict['loss'] = '{:.1e}'.format(loss)
        if proto_net is not None and dataiter_fs is not None:
          post_fix_dict['fs_acc_v'] = '{:.3f}'.format(rtest_fs['acc'] * 100.0)
        it.set_postfix(**post_fix_dict)


def log_results(results, prefix=None, filename=None):
  """Log results to a file."""
  acc = results['acc'] * 100.0
  se = results['acc_se'] * 100.0
  name = prefix if prefix is not None else 'Acc'
  if filename is not None:
    with open(filename, 'a') as f:
      f.write('{}\t\t{:.3f}\t\t{:.3f}\n'.format(name, acc, se))
  log.info(u'{} Acc = {:.3f} ± {:.3f}'.format(name, acc, se))


def main():
  assert tf.executing_eagerly()
  assert args.config is not None, 'You need to pass in model config file path'
  # assert args.data is not None, 'You need to pass in data config file path'
  assert args.env is not None, 'You need to pass in the env config file path'
  assert args.tag is not None, 'You need to specify a tag'
  print(args)
  config = get_config(args.config, ExperimentConfig)
  if args.data is not None:
    data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  model = build_pretrain_net(config)
  # config.model_class = "proto_net"
  # proto_net = build_fewshot_net(config, backbone=model.backbone)
  dataset = get_data(env_config)
  data = get_dataiter(
      dataset,
      config.optimizer_config.batch_size,
      nchw=model.backbone.config.data_format == 'NCHW',
      data_aug=True)

  if args.data is not None:
    config.model_class = "proto_net"
    config.num_classes = data_config.nway
    proto_net = build_fewshot_net(config, backbone=model.backbone)
    dataset_fs = get_data_fs(env_config, load_train=False)
    if dataset_fs['train_fs'] is None and dataset_fs['val_fs'] is not None:
      dataset_fs['train_fs'] = dataset['train']
    data_fs = get_dataiter_fewshot(
        dataset_fs,
        data_config,
        nchw=model.backbone.config.data_format == 'NCHW')
  else:
    data_fs = {'train_fs': None, 'val_fs': None}
    proto_net = None
  save_folder = os.path.join(env_config.results, env_config.dataset, args.tag)

  ckpt_path = env_config.checkpoint
  if len(ckpt_path) > 0 and os.path.exists(ckpt_path):
    ckpt_folder = os.path.join(ckpt_path, os.environ['SLURM_JOB_ID'])
    log.info('Checkpoint folder: {}'.format(ckpt_folder))
  else:
    ckpt_folder = save_folder

  latest = None
  reload_flag = None
  if os.path.exists(ckpt_folder):
    latest = latest_file(ckpt_folder, 'weights-')

  if latest is None and os.path.exists(save_folder):
    latest = latest_file(save_folder, 'weights-')

  if not os.path.exists(save_folder):
    save_config(config, save_folder)

  if latest is not None:
    log.info('Checkpoint already exists. Loading from {}'.format(latest))
    model.load(latest)  # Not loading optimizer weights here.
    reload_flag = latest
    restore_steps = int(reload_flag.split('-')[-1].split('.')[0])
    model.step.assign(restore_steps)

  if not args.eval:
    train(
        model,
        data['train'],
        None,
        save_folder,
        proto_net=proto_net,
        dataiter_fs=data_fs['train_fs'],
        dataiter_test_fs=data_fs['val_fs'],
        reload_flag=reload_flag)
  else:
    # Load the most recent checkpoint.
    model.load(latest_file(save_folder, 'weights-'))

  logfile = os.path.join(save_folder, 'results.tsv')
  data['train'].sampler._cycle = False
  # for split, name in zip(['train', 'val', 'test'], ['Train', 'Val', 'Test']):
  #   if split in data and data[split] is not None:
  #     data[split].reset()
  #     r = evaluate(model, data[split], -1)
  #     log_results(r, prefix=name, filename=logfile)

  if data_fs['val_fs'] is not None:
    for split, name in zip(['train_fs', 'val_fs', 'test_fs'],
                           ['Train', 'Val', 'Test']):
      data_fs[split].reset()
      r_fs = evaluate_fs(proto_net, data_fs[split], 2000)
      log_results(
          r_fs,
          prefix=name + ' {}-Shot'.format(data_config.nshot_max),
          filename=logfile)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pretrain')
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--tag', type=str, default=None)
  args = parser.parse_args()
  main()
