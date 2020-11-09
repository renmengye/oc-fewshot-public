"""Metrics.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import sklearn.metrics


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
  else:
    raise NotImplementedError()
  return np.sum(topk_choices == np.expand_dims(label, axis), axis=axis).mean()


def stderr(array, axis=0):
  """Calculates standard error."""
  if len(array) > 0:
    return array.std(axis=axis) / np.sqrt(float(array.shape[0]))
  else:
    return 0.0


def mean(array, axis=0):
  """Calculates standard error."""
  return array.mean(axis=axis) if len(array) > 0 else 0.0


def calc_ap(results_list, verbose=True):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  y_gt_list = []
  y_full_list = []
  pred_list = []
  score_list = []
  cat_list = []

  for r in results_list:
    flag = r['flag'].astype(np.bool)
    flag_ = np.expand_dims(flag, -1)
    gt_ = r['y_gt'][flag]
    pred_ = np.argmax(r['pred'][:, :, :-1], axis=-1)[flag]
    score_ = r['pred'][:, :, -1][flag]  # Unknown score.

    y_gt_list.append(gt_)
    pred_list.append(pred_)
    score_list.append(score_)  # Category agnostic

    if verbose:
      print('y_gt', y_gt_list[-1], y_gt_list[-1].shape)
      print('pred', pred_list[-1], pred_list[-1].shape)
  y_gt = np.concatenate(y_gt_list)
  score = np.concatenate(score_list)
  y_pred = np.concatenate(pred_list)

  N = len(y_gt)
  sortidx = np.argsort(score)
  score = score[sortidx]
  y_gt = y_gt[sortidx]
  y_pred = y_pred[sortidx]

  tp = (y_gt == y_pred).astype(np.float64)
  pos = (y_gt < unk_id).astype(np.float64)
  npos = pos.sum()

  if verbose:
    print('score sorted', score)
    print('y_gt', y_gt)
    print('y_pred', y_pred)
    print('tp', tp)
    print('unk id', unk_id)

  recall = np.zeros([N], dtype=np.float64)
  tp_cumsum = np.cumsum(tp)
  if verbose:
    print('npos', npos)
    print('tp cumsum', tp_cumsum)
  precision = tp_cumsum / np.arange(1, N + 1).astype(np.float64)
  recall = tp_cumsum / npos
  precision = np.concatenate([[1.0], precision])
  recall = np.concatenate([[0.0], recall])
  ap = sklearn.metrics.auc(recall, precision)
  if verbose:
    print('precision', precision)
    print('recall', recall)
    print('ap', ap)
  return ap


def calc_interval(y_gt, y=None):
  if y is None:
    y = y_gt
  B = y_gt.shape[0]
  # Last time we have seen a class.
  last_seen = np.zeros([B, y_gt.max() + 1]) - 1
  ninterval = np.zeros(y.shape, dtype=np.int64)
  for i in range(y.shape[1]):
    last_seen_ = last_seen[np.arange(B), y_gt[:, i]]
    ninterval[:, i] = i - last_seen_
    last_seen[np.arange(B), y_gt[:, i]] = i
  return ninterval


def calc_nshot(y_gt, y=None):
  if y is None:
    y = y_gt
  nway = np.max(y_gt)
  B, T = y_gt.shape
  waylist = np.arange(nway + 1)
  onehot_bool = np.expand_dims(y_gt, -1) == waylist
  onehot_bool_y = np.expand_dims(y, -1) == waylist
  onehot = (onehot_bool).astype(np.int64)  # [B, T, K]
  onehot_cumsum = np.cumsum(onehot, axis=1)
  nshot = onehot_cumsum[onehot_bool_y].reshape([B, T]) - (y_gt == y).astype(
      np.int64)
  return nshot


def calc_nshot_ap(results_list, nshot_max):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  nshot_list = [calc_nshot(r['y_full']) for r in results_list]
  ap_list = [0.0] * nshot_max

  for n in range(1, nshot_max + 1):
    sel_list = [s == n for s in nshot_list]
    y_gt_list = [r['y_gt'][s][None, :] for s, r in zip(sel_list, results_list)]
    pred_list = [
        r['pred'][s][None, :, :] for s, r in zip(sel_list, results_list)
    ]
    flag_list = [r['flag'][s][None, :] for s, r in zip(sel_list, results_list)]
    subresults = [{
        'y_gt': y,
        'pred': p,
        'flag': f
    } for y, p, f in zip(y_gt_list, pred_list, flag_list)]
    ap_list[n - 1] = calc_ap(subresults, verbose=False)
  return np.array(ap_list)


def calc_acc(results_list):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  y_gt_list = []
  pred_list = []
  acc_list = []

  for r in results_list:
    flag = r['flag'].astype(np.bool)
    y_gt_list.append(r['y_gt'][flag])
    flag_ = np.expand_dims(flag, -1)
    pred_list.append(np.argmax(r['pred'][:, :, :-1], axis=-1)[flag])
    if len(y_gt_list[-1]) > 0:
      acc_list.append(
          np.mean((y_gt_list[-1] == pred_list[-1]).astype(np.float64)))
  y_gt = np.concatenate(y_gt_list)
  y_pred = np.concatenate(pred_list)
  correct = (y_pred == y_gt).astype(np.float64)
  return correct.mean(), stderr(np.array(acc_list))


def calc_nshot_acc(results_list, nshot_max, labeled=False):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  if labeled:
    nshot_list = [calc_nshot(r['y_s'], y=r['y_full']) for r in results_list]
  else:
    nshot_list = [calc_nshot(r['y_full']) for r in results_list]
  acc_list = [0.0] * nshot_max
  stderr_list = [0.0] * nshot_max
  for n in range(1, nshot_max + 1):
    sel_list = [s == n for s in nshot_list]
    known_list = [r['y_gt'] < unk_id for r in results_list]
    sel_list = [np.logical_and(s, k) for s, k in zip(sel_list, known_list)]
    y_gt_list = [r['y_gt'][s][None, :] for s, r in zip(sel_list, results_list)]
    pred_list = [
        r['pred'][s][None, :, :] for s, r in zip(sel_list, results_list)
    ]
    flag_list = [r['flag'][s][None, :] for s, r in zip(sel_list, results_list)]
    subresults = [{
        'y_gt': y,
        'pred': p,
        'flag': f
    } for y, p, f in zip(y_gt_list, pred_list, flag_list)]
    acc_list[n - 1], stderr_list[n - 1] = calc_acc(subresults)
  return np.array(acc_list), np.array(stderr_list)


def calc_nshot_acc_2d(results_list, nappear_max, nshot_max):
  """Combining labeled and unlabeled. X-axis number of appearances, Y-axis
  number of labels."""
  N = nappear_max
  M = nshot_max
  unk_id = results_list[0]['pred'].shape[-1] - 1
  acc_list = np.zeros([N, M])
  stderr_list = np.zeros([N, M])
  nappear_list = [calc_nshot(r['y_full']) for r in results_list]
  nshot_list = [calc_nshot(r['y_s'], y=r['y_full']) for r in results_list]
  for n in range(1, N + 1):
    for m in range(1, M + 1):
      sel_list = [
          np.logical_and(nappear_ == n, nshot_ == m)
          for nappear_, nshot_ in zip(nappear_list, nshot_list)
      ]
      if m > n:
        assert all([np.logical_not(s).all() for s in sel_list])
      known_list = [r['y_gt'] < unk_id for r in results_list]
      sel_list = [np.logical_and(s, k) for s, k in zip(sel_list, known_list)]
      y_gt_list = [
          r['y_gt'][s][None, :] for s, r in zip(sel_list, results_list)
      ]
      pred_list = [
          r['pred'][s][None, :, :] for s, r in zip(sel_list, results_list)
      ]
      flag_list = [
          r['flag'][s][None, :] for s, r in zip(sel_list, results_list)
      ]
      subresults = [{
          'y_gt': y,
          'pred': p,
          'flag': f
      } for y, p, f in zip(y_gt_list, pred_list, flag_list)]
      acc_list[n - 1, m - 1], stderr_list[n - 1, m - 1] = calc_acc(subresults)
  return acc_list, stderr_list


def calc_nshot_acc_3d(results_list, nappear_max, nshot_max, ninterval_split):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  N = nappear_max
  M = nshot_max
  K = len(ninterval_split) - 1
  acc_list = np.zeros([N, M, K])
  stderr_list = np.zeros([N, M, K])
  nappear_list = [calc_nshot(r['y_full']) for r in results_list]
  nshot_list = [calc_nshot(r['y_s'], y=r['y_full']) for r in results_list]
  ninterval_list = [calc_interval(r['y_full']) for r in results_list]
  for n in range(1, N + 1):
    for m in range(1, M + 1):
      for k in range(1, K + 1):
        sel_list = [
            np.logical_and(
                np.logical_and(nappear_ == n, nshot_ == m),
                np.logical_and(ninterval_ >= ninterval_split[k - 1],
                               ninterval_ < ninterval_split[k])) for nappear_,
            nshot_, ninterval_ in zip(nappear_list, nshot_list, ninterval_list)
        ]
        if m > n:
          assert all([np.logical_not(s).all() for s in sel_list])

        known_list = [r['y_gt'] < unk_id for r in results_list]
        sel_list = [np.logical_and(s, k) for s, k in zip(sel_list, known_list)]
        y_gt_list = [
            r['y_gt'][s][None, :] for s, r in zip(sel_list, results_list)
        ]
        pred_list = [
            r['pred'][s][None, :, :] for s, r in zip(sel_list, results_list)
        ]
        flag_list = [
            r['flag'][s][None, :] for s, r in zip(sel_list, results_list)
        ]
        subresults = [{
            'y_gt': y,
            'pred': p,
            'flag': f
        } for y, p, f in zip(y_gt_list, pred_list, flag_list)]
        acc_list[n - 1, m - 1, k - 1], stderr_list[n - 1, m - 1, k -
                                                   1] = calc_acc(subresults)
  return acc_list, stderr_list


def calc_nshot_acc_2dv2(results_list, nappear_max, ninterval_split):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  N = nappear_max
  K = len(ninterval_split) - 1
  acc_list = np.zeros([N, K])
  stderr_list = np.zeros([N, K])
  nappear_list = [calc_nshot(r['y_full']) for r in results_list]
  nshot_list = [calc_nshot(r['y_s'], y=r['y_full']) for r in results_list]
  ninterval_list = [calc_interval(r['y_full']) for r in results_list]
  # print(ninterval_list[0])
  for n in range(1, N + 1):
    for k in range(1, K + 1):
      sel_list = [
          np.logical_and(
              nappear_ == n,
              np.logical_and(ninterval_ >= ninterval_split[k - 1],
                             ninterval_ < ninterval_split[k]))
          for nappear_, ninterval_ in zip(nappear_list, ninterval_list)
      ]
      known_list = [r['y_gt'] < unk_id for r in results_list]
      sel_list = [np.logical_and(s, k) for s, k in zip(sel_list, known_list)]
      y_gt_list = [
          r['y_gt'][s][None, :] for s, r in zip(sel_list, results_list)
      ]
      pred_list = [
          r['pred'][s][None, :, :] for s, r in zip(sel_list, results_list)
      ]
      flag_list = [
          r['flag'][s][None, :] for s, r in zip(sel_list, results_list)
      ]
      subresults = [{
          'y_gt': y,
          'pred': p,
          'flag': f
      } for y, p, f in zip(y_gt_list, pred_list, flag_list)]
      acc_list[n - 1, k - 1], stderr_list[n - 1, k - 1] = calc_acc(subresults)
  return acc_list, stderr_list


def calc_nshot_acc_2dv3(results_list, nshot_max, ninterval_split):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  M = nshot_max
  K = len(ninterval_split) - 1
  acc_list = np.zeros([M, K])
  stderr_list = np.zeros([M, K])
  nappear_list = [calc_nshot(r['y_full']) for r in results_list]
  nshot_list = [calc_nshot(r['y_s'], y=r['y_full']) for r in results_list]
  ninterval_list = [calc_interval(r['y_full']) for r in results_list]
  for m in range(1, M + 1):
    for k in range(1, K + 1):
      sel_list = [
          np.logical_and(
              nshot_ == m,
              np.logical_and(ninterval_ >= ninterval_split[k - 1],
                             ninterval_ < ninterval_split[k]))
          for nshot_, ninterval_ in zip(nshot_list, ninterval_list)
      ]
      known_list = [r['y_gt'] < unk_id for r in results_list]
      sel_list = [np.logical_and(s, k) for s, k in zip(sel_list, known_list)]
      y_gt_list = [
          r['y_gt'][s][None, :] for s, r in zip(sel_list, results_list)
      ]
      pred_list = [
          r['pred'][s][None, :, :] for s, r in zip(sel_list, results_list)
      ]
      flag_list = [
          r['flag'][s][None, :] for s, r in zip(sel_list, results_list)
      ]
      subresults = [{
          'y_gt': y,
          'pred': p,
          'flag': f
      } for y, p, f in zip(y_gt_list, pred_list, flag_list)]
      acc_list[m - 1, k - 1], stderr_list[m - 1, k - 1] = calc_acc(subresults)
  return acc_list, stderr_list


def calc_acc_time(results_list, tmax):
  acc_time = []  # [T, N] T=number of timestep; N=number of episodes
  for t in range(tmax):
    acc_time.append([])
  for i, r in enumerate(results_list):
    # Support set metrics, accumulate per time step.
    correct = label_equal(r['pred_id'], r['y_gt'])  # [B, T]
    for t in range(tmax):
      if r['flag'][:, t].sum() > 0:
        acc_time[t].append(correct[:, t].sum() / r['flag'][:, t].sum())
  acc_time = [np.array(l) for l in acc_time]
  return np.array([mean(l) for l in acc_time]), np.array(
      [stderr(l) for l in acc_time])


def calc_acc_time_label(results_list, tmax):
  unk_id = results_list[0]['pred'].shape[-1] - 1
  acc_time = []  # [T, N] T=number of timestep; N=number of episodes
  for t in range(tmax):
    acc_time.append([])
  for i, r in enumerate(results_list):
    # Support set metrics, accumulate per time step.
    correct = label_equal(np.argmax(r['pred'][:, :, :-1], axis=-1),
                          r['y_gt'])  # [B, T]
    T = r['y_gt'].shape[1]
    flag = r['flag']
    is_unk = (r['y_gt'] == unk_id).astype(np.float32)  # [B, T]
    flag = flag * (1.0 - is_unk)
    for t in range(tmax):
      if flag[:, t].sum() > 0:
        acc_time[t].append(correct[:, t].sum())
  acc_time = [np.array(l) for l in acc_time]
  return np.array([mean(l) for l in acc_time]), np.array(
      [stderr(l) for l in acc_time])


if __name__ == '__main__':
  y_s = np.array([[1, 10, 3, 2, 10, 2, 3, 1, 2, 2, 2, 2]])
  y_full = np.array([[1, 2, 3, 2, 2, 2, 3, 1, 2, 2, 2, 2]])
  y_gt = np.array([[10, 10, 10, 10, 2, 2, 3, 1, 2, 2, 2, 2]])
  pred = np.array([[10, 10, 10, 10, 2, 2, 3, 1, 2, 3, 2, 2]])
  flag = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
  pred2 = np.zeros([1, y_s.shape[1], 11])
  pred2[np.zeros([y_s.shape[1]], dtype=y_s.dtype),
        np.arange(y_s.shape[1]), pred[0]] = 1.0
  print(pred2)
  results_list = [{
      'y_s': y_s,
      'y_gt': y_gt,
      'y_full': y_full,
      'pred': pred2,
      'flag': flag
  }]
  print(calc_nshot_acc(results_list, nshot_max=5, labeled=True))
  print(calc_nshot_acc(results_list, nshot_max=5))
  print(calc_ap(results_list, verbose=True))
  print(calc_nshot_acc_2d(results_list, nappear_max=5, nshot_max=5))
  print('interval', calc_interval(y_full))
  acc, se = calc_acc_time_label(results_list, tmax=12)
  print(acc)
