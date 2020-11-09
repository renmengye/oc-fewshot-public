from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

DATASET_REGISTRY = {}
ITERATOR_REGISTRY = {}
SAMPLER_REGISTRY = {}


def RegisterDataset(name):
  """Registers a dataset class"""

  def decorator(f):
    DATASET_REGISTRY[name] = f
    return f

  return decorator


def RegisterSampler(name):
  """Registers a sampler class"""

  def decorator(f):
    SAMPLER_REGISTRY[name] = f
    return f

  return decorator


def RegisterIterator(name):
  """Registers an iterator class."""

  def decorator(f):
    ITERATOR_REGISTRY[name] = f

    return f

  return decorator
