from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from fewshot.utils.logger import get as get_logger

log = get_logger()


class VariableManager():

  def __init__(self):
    self.scope_list = []
    self.var_dict = {}

  def enter_scope(self, scope):
    self.scope_list.append(scope)

  def exit_scope(self):
    self.scope_list.pop(-1)

  def get_variable(self, name, initializer, **kwargs):
    pref = self.get_prefix()
    if len(pref) > 0:
      fullname = self.get_prefix() + '/' + name
      # fullname = self.get_prefix() + '_' + name
    else:
      fullname = name
    if "wdict" in kwargs and kwargs["wdict"] is not None:
      if fullname in kwargs["wdict"]:
        log.info('Shared using wdict: {}'.format(fullname))
        return kwargs["wdict"][fullname]
    if "wdict" in kwargs:
      del kwargs["wdict"]
    if fullname in self.var_dict:
      return self.var_dict[fullname]
    else:
      var = tf.Variable(initializer(), name=fullname, **kwargs)
      self.var_dict[fullname] = var
      return var

  def set_variable(self, var_old, var):
    found = False
    for k in self.var_dict:
      if self.var_dict[k] is var_old:
        self.var_dict[k] = var
        found = True
        break
    if not found:
      assert False, var_old.name

  def set_collection(self, collection):
    pass

  def get_prefix(self):
    slist = []
    for s in self.scope_list:
      slist.append(s.name)
    return '/'.join(slist)
    # return '_'.join(slist)

  def reset(self):
    self.var_dict.clear()


var_mgr = VariableManager()


class VariableScope():

  def __init__(self, name):
    self._name = name

  def __enter__(self):
    var_mgr.enter_scope(self)
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    var_mgr.exit_scope()

  @property
  def name(self):
    return self._name


def variable_scope(name):
  return VariableScope(name)


def get_variable(*args, **kwargs):
  return var_mgr.get_variable(*args, **kwargs)


def set_variable(*args, **kwargs):
  return var_mgr.set_variable(*args, **kwargs)


def reset_variables():
  return var_mgr.reset()
