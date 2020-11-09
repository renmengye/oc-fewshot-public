"""Visualize per-timestep accuracy.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pkl

FOLDER = '/mnt/local/results/fewshot-lifelong/omniglot'
RESULTS_PATH_TMPL = os.path.join(FOLDER, '{}', 'results.pkl')
MODEL_ID_DICT = {
    'Online ProtoNet': 'debug_mindist_b32',
    'LSTM': 'debug_lstm_x2_b32_ln',
    # 'MANN': 'debug_mann_ln_x2',
    # 'DNC': 'debug_mann_dnc_layernorm_x2',
    'DNC': 'debug_dnc_sigmoidunk_1.0_2',
    'DNC-AP': 'debug_dnc_apv2_ap20_sigmoid_80k100k120k'
}
CMAP = ['IndianRed', 'LightCoral', 'LightBlue', 'SteelBlue']


def plot_time_acc(data):
  fig = plt.figure()
  for i, k in enumerate(data.keys()):
    plt.plot(np.arange(len(data[k])), data[k], color=CMAP[i])

  plt.xlabel('Time Step')
  plt.ylabel('Acc')
  plt.legend(data.keys())
  plt.savefig('output/time_acc.pdf')


def plot_time_acc_seaborn_old(data, data_se):
  sns.set()
  fig = plt.figure(figsize=(9, 5))
  models = list(data.keys())
  T = len(data[models[0]])
  M = len(models)
  timestep = np.arange(T)
  dfdata = {}
  dfdata['Time'] = np.tile(timestep.reshape([1, -1]), [M, 1]).reshape([-1])
  dfdata['Model'] = np.tile(np.array(models).reshape([-1, 1]),
                            [1, T]).reshape([-1])
  dfdata['Acc'] = np.concatenate([data[m] for m in models])
  dfdata['Acc_SE'] = np.concatenate([data_se[m] for m in models])
  print(dfdata['Acc'].shape)
  print(dfdata['Acc_SE'].shape)
  print(dfdata['Model'].shape)
  print(dfdata['Time'].shape)
  df = pd.DataFrame(data=dfdata)
  sns.lineplot(
      x='Time',
      y='Acc',
      hue='Model',
      data=df,
      palette=CMAP,
      linewidth=3,
      ci='Acc_SE')
  ax = plt.gca()
  ax.set_xlabel('Time Steps', fontsize=18)
  ax.set_ylabel('Acc. (%)', fontsize=18)
  ax.set_xticklabels(ax.get_xticks(), {'fontsize': 16})
  ax.set_yticklabels(ax.get_yticks(), {'fontsize': 16})
  labels = [item.get_text() for item in ax.get_xticklabels()]
  print(labels)
  new_labels = ["%d" % int(float(l)) if '.0' in l else '' for l in labels]
  ax.set_xticklabels(new_labels)
  handles, labels = plt.gca().get_legend_handles_labels()
  plt.legend(handles=handles[1:], labels=labels[1:], fontsize=16, loc=3)
  plt.tight_layout()
  plt.savefig('output/time_acc.pdf')


def plot_time_acc_seaborn(data, fname):
  sns.set()
  fig = plt.figure(figsize=(9, 5))
  models = list(data.keys())
  T = len(data[models[0]])
  M = len(models)
  timestep = np.arange(T)
  dfdata = {'Time': [], 'Acc': [], 'Model': []}
  for m in models:
    acc_time_data = data[m]['test_fs']['acc_time_all']
    for i in range(len(acc_time_data)):
      entry = acc_time_data[i]
      for j in range(len(entry)):
        dfdata['Time'].append(i)
        dfdata['Acc'].append(entry[j])
        dfdata['Model'].append(m)
  for k in dfdata.keys():
    dfdata[k] = np.array(dfdata[k])
  df = pd.DataFrame(data=dfdata)
  sns.lineplot(
      x='Time', y='Acc', hue='Model', data=df, palette=CMAP, linewidth=2)
  ax = plt.gca()
  ax.set_xlabel('Time Steps', fontsize=18)
  ax.set_ylabel('Acc. (%)', fontsize=18)
  ax.set_xticklabels(ax.get_xticks(), {'fontsize': 16})
  ax.set_yticklabels(ax.get_yticks(), {'fontsize': 16})
  xlabels = [item.get_text() for item in ax.get_xticklabels()]
  ylabels = [item.get_text() for item in ax.get_yticklabels()]
  new_xlabels = ["%d" % int(float(l)) if '.0' in l else '' for l in xlabels]
  new_ylabels = ["%d" % int(float(l) * 100.0) for l in ylabels]
  ax.set_xticklabels(new_xlabels)
  ax.set_yticklabels(new_ylabels)
  handles, labels = plt.gca().get_legend_handles_labels()
  plt.legend(handles=handles[1:], labels=labels[1:], fontsize=16, loc=3)
  # Indicate training steps.
  plt.axvline(x=40, color='k', linestyle='--')
  plt.tight_layout()
  for ext in ['.pdf', '.png']:
    plt.savefig(fname + ext)
    print('Output saved to {}{}'.format(fname, ext))


def read_data(results_path):
  results_dict = {'val': {}, 'test': {}}

  with open(results_path, 'r') as f:
    for line in f:
      parts = line.split(',')

      # Val or test.
      if line.startswith('Val'):
        results = results_dict['val']
        continue
      elif line.startswith('Test'):
        results = results_dict['test']
        continue

      # Make an array.
      results[parts[0]] = np.array([float(p) for p in parts[1:]])
  print(results_dict)
  return results_dict


def main_old():
  plot_data = {}
  plot_data_se = {}
  for k in MODEL_ID_DICT.keys():
    data = read_data(RESULTS_PATH_TMPL.format(MODEL_ID_DICT[k]))
    print(k)
    print(data)
    plot_data[k] = data['val']['acc time']
    plot_data_se[k] = data['val']['acc time se']
  plot_time_acc_seaborn_old(plot_data, plot_data_se)


def main():
  plot_data = {}
  for k in MODEL_ID_DICT.keys():
    data = pkl.load(open(RESULTS_PATH_TMPL.format(MODEL_ID_DICT[k]), 'rb'))
    plot_data[k] = data
  output_fname = './output/time_acc'
  plot_time_acc_seaborn(plot_data, output_fname)


if __name__ == '__main__':
  main()
