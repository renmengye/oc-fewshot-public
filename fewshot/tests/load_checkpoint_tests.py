"""Test loading the checkpoints.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.get_stats import get_stats
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.experiments.oc_fewshot import evaluate
from fewshot.utils.logger import get as get_logger
from fewshot.models.variable_context import reset_variables

log = get_logger()


def test_load_one(folder, seed=0):
  # log.info('Command line args {}'.format(args))
  config_file = os.path.join(folder, 'config.prototxt')
  config = get_config(config_file, ExperimentConfig)
  # config.c4_config.data_format = 'NHWC'
  # config.resnet_config.data_format = 'NHWC'
  if 'omniglot' in folder:
    if 'ssl' in folder:
      data_config_file = 'configs/episodes/roaming-omniglot/roaming-omniglot-150-ssl.prototxt'  # NOQA
    else:
      data_config_file = 'configs/episodes/roaming-omniglot/roaming-omniglot-150.prototxt'  # NOQA
    env_config_file = 'configs/environ/roaming-omniglot-docker.prototxt'
  elif 'rooms' in folder:
    if 'ssl' in folder:
      data_config_file = 'configs/episodes/roaming-rooms/roaming-rooms-100.prototxt'  # NOQA
    else:
      data_config_file = 'configs/episodes/roaming-rooms/romaing-rooms-100.prototxt'  # NOQA
    env_config_file = 'configs/environ/roaming-rooms-docker.prototxt'
  data_config = get_config(data_config_file, EpisodeConfig)
  env_config = get_config(env_config_file, EnvironmentConfig)
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

  model = build_pretrain_net(config)
  mem_model = build_net(config, backbone=model.backbone)
  reload_flag = None
  restore_steps = 0

  # Reload previous checkpoint.
  latest = latest_file(folder, 'best-')
  if latest is None:
    latest = latest_file(folder, 'weights-')
  assert latest is not None, "Checkpoint not found."

  if latest is not None:
    log.info('Checkpoint already exists. Loading from {}'.format(latest))
    mem_model.load(latest)  # Not loading optimizer weights here.
    reload_flag = latest
    restore_steps = int(reload_flag.split('-')[-1])

  # Get dataset.
  dataset = get_data_fs(env_config, load_train=True)

  # Get data iterators.
  if env_config.dataset in ["roaming-rooms", "matterport"]:
    data = get_dataiter_sim(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=mem_model.backbone.config.data_format == 'NCHW',
        seed=seed + restore_steps)
  else:
    data = get_dataiter_continual(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=mem_model.backbone.config.data_format == 'NCHW',
        save_additional_info=True,
        random_box=data_config.random_box,
        seed=seed + restore_steps)

  # Load the most recent checkpoint.
  latest = latest_file(folder, 'best-')
  if latest is None:
    latest = latest_file(folder, 'weights-')

  data['trainval_fs'].reset()
  data['val_fs'].reset()
  data['test_fs'].reset()

  results_all = {}
  split_list = ['trainval_fs', 'val_fs', 'test_fs']
  name_list = ['Train', 'Val', 'Test']
  nepisode_list = [5,5,5]
  # nepisode_list = [600, config.num_episodes, config.num_episodes]

  for split, name, N in zip(split_list, name_list, nepisode_list):
    # print(name)
    data[split].reset()
    r1 = evaluate(mem_model, data[split], N, verbose=False)
    stats = get_stats(r1, tmax=data_config.maxlen)
    print(split, stats['ap'])
    # log_results(stats, prefix=name, filename=logfile)


def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  with log.verbose_level(0):
    for ln in open(os.path.join(dir_path, 'checkpoints.txt'), 'r').readlines():
      reset_variables()
      print(ln)
      folder = ln.strip('\n')
      test_load_one(folder)


if __name__ == '__main__':
  main()
