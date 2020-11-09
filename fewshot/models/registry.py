from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

MODEL_REGISTRY = {}
MODULE_REGISTRY = {}


def RegisterModel(model_name):
  """Registers a model class"""

  def decorator(f):
    MODEL_REGISTRY[model_name] = f
    return f

  return decorator


def RegisterModule(module_name):
  """Registers a model class"""

  def decorator(f):
    MODULE_REGISTRY[module_name] = f
    return f

  return decorator
