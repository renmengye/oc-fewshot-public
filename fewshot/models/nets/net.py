"""A standalone network.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import pickle as pkl

from collections import OrderedDict

from fewshot.models.modules.container_module import ContainerModule
from fewshot.utils.logger import get as get_logger

log = get_logger()


class Net(ContainerModule):

  def __init__(self, dtype=tf.float32):
    super(Net, self).__init__(dtype=dtype)
    self._var_to_optimize = None
    self._regularized_weights = None
    self._regularizer_wd = None

  def train_step(self, *args, **kwargs):
    raise NotImplementedError()

  def eval_step(self, *args, **kwargs):
    raise NotImplementedError()

  def _get_optimizer(self, optname, learn_rate):
    """Gets an optimizer.

    Args:
      optname: String. Name of the optimizer.
    """
    clip_norm = self.config.optimizer_config.clip_norm
    # assert clip_norm > 1.0, str(clip_norm)
    if clip_norm < 0.001:
      kwargs = {}
    else:
      kwargs = {'clipnorm': clip_norm}
    if optname == 'adam':
      opt = tf.optimizers.Adam(learning_rate=learn_rate, **kwargs)
    elif optname == 'momentum':
      opt = tf.optimizers.SGD(learning_rate=learn_rate, momentum=0.9, **kwargs)
    elif optname == 'nesterov':
      opt = tf.optimizers.SGD(
          learning_rate=learn_rate, momentum=0.9, use_nesterov=True, **kwargs)
    else:
      raise ValueError('Unknown optimizer {}'.format(optname))
    return opt

  def save(self, path):
    """Saves the weights."""
    # log.info('Save to {}'.format(path))
    wdict = OrderedDict()
    items = [(w.name, w.numpy()) for w in self.weights()]
    optimizer_items = self._optimizer.get_weights()
    for item in items:
      assert item[0] not in wdict, 'Key exists: {}'.format(item[0])
      wdict[item[0]] = item[1]

    wdict['__optimizer__'] = optimizer_items
    # np.savez(path, **wdict)
    pkl.dump(wdict, open(path, 'wb'))

  def load(self, path, load_optimizer=False):
    """Loads the weights."""
    print(path)
    log.info('Restore from {}'.format(path))
    wdict = OrderedDict([(w.name, w) for w in self.weights()])
    [log.info('Weights {} {}'.format(w.name, w.shape)) for w in self.weights()]
    loaddict = pkl.load(open(path, 'rb'))
    if '__optimizer__' in loaddict and load_optimizer:
      self._optimizer.set_weights(loaddict['__optimizer__'])

    if '__optimizer__' in loaddict:
      del loaddict['__optimizer__']
    for k in loaddict.keys():
      if k in wdict:
        log.info('Loaded {} {}'.format(k, wdict[k].shape))
        wdict[k].assign(loaddict[k])
      else:
        log.error('Variable {} not found in the current graph'.format(k))

  def get_var_to_optimize(self):
    """Gets the list of variables to optimize."""
    var = self.weights()
    var = list(filter(lambda x: x.trainable, var))
    return var

  def var_to_optimize(self):
    if self._var_to_optimize is None:
      self._var_to_optimize = self.get_var_to_optimize()
      for v in self._var_to_optimize:
        log.info("Trainable weight {}".format(v.name))
      log.info('Num trainable weight: {}'.format(len(self._var_to_optimize)))
    return self._var_to_optimize

  def _get_regularizer_loss(self, *w_list):
    """Computes L2 loss."""
    if len(w_list) > 0:
      return tf.add_n([tf.reduce_sum(w**2) * 0.5 for w in w_list])
    else:
      return 0.0

  def regularized_weights(self):
    """List of weights to be L2 regularized"""
    if self._regularized_weights is None:
      self._regularized_weights = list(
          filter(lambda x: x.name.endswith('w:0'), self.var_to_optimize()))
      for w in self._regularized_weights:
        log.info("Add weight decay {} {:.3e}".format(w.name, self.wd))
    return self._regularized_weights

  def set_trainable(self, trainable):
    """Set parameter trainable status."""
    self._var_to_optimize = None
    super(Net, self).set_trainable(trainable)

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def wd(self):
    """Weight decay coefficient"""
    raise NotImplementedError()
