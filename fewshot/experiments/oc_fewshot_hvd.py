"""
Train an online few-shot network.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import horovod.tensorflow as hvd
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(
      gpus[hvd.local_rank() % len(gpus)], 'GPU')
is_chief = hvd.rank() == 0

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.oc_fewshot import train, dummy_context_mgr
from fewshot.experiments.utils import ExperimentLogger
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.experiments.utils import save_config
from fewshot.utils.logger import get as get_logger

log = get_logger()


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

  # Modify optimization config.
  if config.optimizer_config.lr_scaling:
    for i in range(len(config.optimizer_config.lr_decay_steps)):
      config.optimizer_config.lr_decay_steps[i] //= len(gpus)
    config.optimizer_config.max_train_steps //= len(gpus)

    # Linearly scale learning rate.
    for i in range(len(config.optimizer_config.lr_list)):
      config.optimizer_config.lr_list[i] *= float(len(gpus))

  log.info('Number of classes {}'.format(data_config.nway))

  # Build model.
  model = build_pretrain_net(config)
  mem_model = build_net(config, backbone=model.backbone, distributed=True)
  reload_flag = None
  restore_steps = 0

  if 'SLURM_JOB_ID' in os.environ:
    log.info('SLURM job ID: {}'.format(os.environ['SLURM_JOB_ID']))

  # Create save folder.
  if is_chief:
    save_folder = os.path.join(env_config.results, env_config.dataset,
                               args.tag)
    ckpt_path = env_config.checkpoint
    if len(ckpt_path) > 0 and os.path.exists(ckpt_path):
      ckpt_folder = os.path.join(ckpt_path, os.environ['SLURM_JOB_ID'])
      log.info('Checkpoint folder: {}'.format(ckpt_folder))
    else:
      ckpt_folder = save_folder

    latest = None
    if os.path.exists(ckpt_folder):
      latest = latest_file(ckpt_folder, 'weights-')

    if latest is None and os.path.exists(save_folder):
      latest = latest_file(save_folder, 'weights-')

    if latest is not None:
      log.info('Checkpoint already exists. Loading from {}'.format(latest))
      mem_model.load(latest)  # Not loading optimizer weights here.
      reload_flag = latest
      restore_steps = int(reload_flag.split('-')[-1])

    # Create TB logger.
    save_config(config, save_folder)
    writer = tf.summary.create_file_writer(save_folder)
    logger = ExperimentLogger(writer)
  else:
    save_folder = None
    ckpt_folder = None
    writer = None
    logger = None

  # Get dataset.
  dataset = get_data_fs(env_config, load_train=True)

  # Get data iterators.
  if env_config.dataset in ["matterport", "roaming-rooms"]:
    data = get_dataiter_sim(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=mem_model.backbone.config.data_format == 'NCHW',
        distributed=True,
        seed=args.seed + restore_steps)
  else:
    data = get_dataiter_continual(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=mem_model.backbone.config.data_format == 'NCHW',
        save_additional_info=True,
        random_box=data_config.random_box,
        distributed=True,
        seed=args.seed + restore_steps)

  # Load model, training loop.
  if args.pretrain is not None and reload_flag is None:
    mem_model.load(latest_file(args.pretrain, 'weights-'))
    if config.freeze_backbone:
      model.set_trainable(False)  # Freeze the network.
      log.info('Backbone network is now frozen')

  with writer.as_default() if writer is not None else dummy_context_mgr(
  ) as gs:
    train(
        mem_model,
        data['train_fs'],
        data['trainval_fs'],
        data['val_fs'],
        ckpt_folder,
        final_save_folder=save_folder,
        maxlen=data_config.maxlen,
        logger=logger,
        writer=writer,
        is_chief=is_chief,
        reload_flag=reload_flag)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Lifelong Few-Shot Training')
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--pretrain', type=str, default=None)
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  tf.random.set_seed(1234)
  main()
