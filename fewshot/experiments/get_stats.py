"""Compute statistics of online few-shot episodes.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import time

from datetime import datetime
from fewshot.utils.logger import get as get_logger
from fewshot.experiments.metrics import calc_ap
from fewshot.experiments.metrics import calc_nshot_acc
from fewshot.experiments.metrics import calc_acc_time
from fewshot.experiments.metrics import calc_acc_time_label
# from fewshot.experiments.metrics import calc_nshot_acc_2d
# from fewshot.experiments.metrics import calc_nshot_acc_3d
# from fewshot.experiments.metrics import calc_nshot_acc_2dv2
# from fewshot.experiments.metrics import calc_nshot_acc_2dv3

log = get_logger()


def get_stats(results,
              nshot_max=5,
              tmax=40,
              use_query=False,
              skip_unk=False,
              distractor=False):
  """Get performance statistics from the list of episodic results."""
  num_steps = len(results)
  acc_query = np.zeros([num_steps])
  # Accuracy at 0-5 shot.
  # Since each episodes are different, here we use a list of list.
  acc_support = []  # [S, N] S=number of shots; N=number of episodes
  acc_query_label = {}  # [K, N] L=number of way; N=number of episodes
  acc_time = []  # [T, N] T=number of timestep; N=number of episodes
  if distractor:
    acc_support_distractor = []  # [N] N=number of episodes
    acc_query_distractor = []  # [N] N=number of episodes
  for s in range(nshot_max):
    acc_support.append([])
  for t in range(tmax):
    acc_time.append([])
  acc_all = []  # [T] Overall accuracy.
  acc_label = []
  acc_unlabel = []
  acc_stage = []
  ap = calc_ap(results, verbose=False)
  acc_nshot, acc_nshot_se = calc_nshot_acc(
      results, nshot_max=nshot_max, labeled=False)
  acc_nshot_labeled, acc_nshot_labeled_se = calc_nshot_acc(
      results, nshot_max=nshot_max, labeled=True)
  # acc_nshot_2d, acc_nshot_2d_se = calc_nshot_acc_2d(
  #     results, nappear_max=nshot_max * 2, nshot_max=nshot_max)
  # acc_nshot_3d, acc_nshot_3d_se = calc_nshot_acc_3d(
  #     results,
  #     nappear_max=5,
  #     nshot_max=3,
  #     ninterval_split=[1, 3, 6, 11, 21, 51, 101])
  # acc_nshot_2dv2, acc_nshot_2dv2_se = calc_nshot_acc_2dv2(
  #     results, nappear_max=5, ninterval_split=[1, 3, 6, 11, 21, 51, 101])
  # acc_nshot_2dv3, acc_nshot_2dv3_se = calc_nshot_acc_2dv3(
  #     results, nshot_max=5, ninterval_split=[1, 3, 6, 11, 21, 51, 101])

  acc_time, acc_time_se = calc_acc_time(results, tmax=tmax)
  acc_time_label, acc_time_label_se = calc_acc_time_label(results, tmax=tmax)
  results_dict = {
      'ap': ap,
      'acc_nshot': acc_nshot,
      'acc_nshot_se': acc_nshot_se,
      'acc_nshot_labeled': acc_nshot_labeled,
      'acc_nshot_labeled_se': acc_nshot_labeled_se,
      # 'acc_nshot_2d': acc_nshot_2d,
      # 'acc_nshot_2d_se': acc_nshot_2d_se,
      # 'acc_nshot_3d': acc_nshot_3d,
      # 'acc_nshot_3d_se': acc_nshot_3d_se,
      # 'acc_nshot_2dv2': acc_nshot_2dv2,
      # 'acc_nshot_2dv2_se': acc_nshot_2dv2_se,
      # 'acc_nshot_2dv3': acc_nshot_2dv3,
      # 'acc_nshot_2dv3_se': acc_nshot_2dv3_se,
      'acc_time': acc_time,
      'acc_time_se': acc_time_se,
      'acc_time_label': acc_time_label,
      'acc_time_label_se': acc_time_label_se
  }
  return results_dict


def log_results(results, prefix, filename=None):
  """Log results to a file."""
  if filename is not None:
    fcsv = open(filename, 'a')
    timestamp = time.time()
    dt = datetime.fromtimestamp(timestamp)
    timestr = dt.strftime('%Y/%m/%d %H:%M:%S')
    fcsv.write(prefix + ' ' + timestr + '\n')
    fcsv.write('ap,{:.3f}\n'.format(results['ap'] * 100.0))
    for name in [
        'acc_nshot', 'acc_nshot_se', 'acc_nshot_labeled',
        'acc_nshot_labeled_se', 'acc_time', 'acc_time_se', 'acc_time_label',
        'acc_time_label_se'
    ]:
      fcsv.write(prefix + ' ' + name + ',' +
                 ','.join(['{:.3f}'.format(x * 100.0)
                           for x in results[name]]) + '\n')

    # for name in ['acc_nshot_2d', 'acc_nshot_2d_se']:
    #   for s in range(results[name].shape[1]):
    #     fcsv.write(prefix + ' ' + name + ',' + ','.join(
    #         ['{:.3f}'.format(x * 100.0) for x in results[name][:, s]]) +
    # '\n')
    fcsv.close()

  log.info('{} AP: {:.3f}'.format(prefix, results['ap'] * 100.0))
  for name, name2 in zip(['acc_nshot', 'acc_nshot_labeled'],
                         ['ACC', 'ACC Labeled']):
    for s in range(len(results[name])):
      log.info(u'{} {} {}-Shot: {:.3f} ± {:.3f}'.format(
          prefix, name2, s + 1, results[name][s] * 100.0,
          results[name + '_se'][s] * 100.0))

  # for name, name2 in zip(['acc_nshot_2d'], ['ACC 2D']):
  #   for s in range(results[name].shape[1]):
  #     info = u'{} {} {}-Shot: '.format(prefix, name2, s + 1)
  #     for a in range(results[name].shape[0]):
  #       info += u' {:.3f} ± {:.3f}'.format(results[name][a, s] * 100.0,
  #                                          results[name + '_se'][a, s]
  # * 100.0)
  #     log.info(info)
  # np.savetxt(
  #     filename + ".{}.3ds1.csv".format(prefix),
  #     results['acc_nshot_3d'][:, 0, :] * 100.0,
  #     delimiter=",")
  # np.savetxt(
  #     filename + ".{}.3ds2.csv".format(prefix),
  #     results['acc_nshot_3d'][:, 1, :] * 100.0,
  #     delimiter=",")
  # np.savetxt(
  #     filename + ".{}.3ds3.csv".format(prefix),
  #     results['acc_nshot_3d'][:, 2, :] * 100.0,
  #     delimiter=",")
  # np.savetxt(
  #     filename + ".{}.2dv2.csv".format(prefix),
  #     results['acc_nshot_2dv2'] * 100.0,
  #     delimiter=",")
  # np.savetxt(
  #     filename + ".{}.2dv3.csv".format(prefix),
  #     results['acc_nshot_2dv3'] * 100.0,
  #     delimiter=",")
