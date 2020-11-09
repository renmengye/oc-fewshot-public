from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import fewshot.models.modules  # NOQA
import fewshot.models.nets  # NOQA
import fewshot.models.registry as registry

from fewshot.utils import logger

log = logger.get()


def get_model(model_name, *args, **kwargs):
  log.info("Model {}".format(model_name))
  if model_name in registry.MODEL_REGISTRY:
    return registry.MODEL_REGISTRY[model_name](*args, **kwargs)
  else:
    raise ValueError("Model class does not exist {}".format(model_name))


def get_module(model_name, *args, **kwargs):
  log.info("Model {}".format(model_name))
  if model_name in registry.MODULE_REGISTRY:
    return registry.MODULE_REGISTRY[model_name](*args, **kwargs)
  else:
    raise ValueError("Module class does not exist {}".format(model_name))
