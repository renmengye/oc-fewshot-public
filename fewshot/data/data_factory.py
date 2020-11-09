from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import fewshot.data.datasets  # NOQA
import fewshot.data.registry as registry

from fewshot.utils import logger

log = logger.get()


def get_dataset(dataset_name, folder, split, *args, **kwargs):
  log.info('Dataset {}'.format(dataset_name))
  if dataset_name in registry.DATASET_REGISTRY:
    return registry.DATASET_REGISTRY[dataset_name](folder, split, *args,
                                                   **kwargs)
  else:
    raise ValueError("Unknown dataset \"{}\"".format(dataset_name))


def get_sampler(sampler_name, seed, *args, **kwargs):
  log.info('Sampler {}'.format(sampler_name))
  if sampler_name in registry.SAMPLER_REGISTRY:
    return registry.SAMPLER_REGISTRY[sampler_name](seed, *args, **kwargs)
  else:
    raise ValueError("Unknown sampler \"{}\"".format(sampler_name))


# def get_concurrent_iterator(dataset, max_queue_size=100, num_threads=10):
#   return ConcurrentBatchIterator(
#       dataset,
#       max_queue_size=max_queue_size,
#       num_threads=num_threads,
#       log_queue=-1)
