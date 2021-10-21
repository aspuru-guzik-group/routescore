#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Custom style
plt.style.use('scientific')

# absolute tolerances for chimera
absolutes = np.array([0.67, 1080000, 0.2, 0.15848931924611134])

# load in gryffin runs with Naive score as objective
df_naive = pd.read_pickle('Optimization/runs/gryffin_runs_naive.pkl')

# make the plot

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 10))

sns.lineplot(x='eval', y='peak_score', data=df_naive, ax=axes[0], label='Naive Score Included')
axes[0].axhline(absolutes[0], ls='--', linewidth=2, c='k', alpha=0.6)
axes[0].fill_between(df_naive['eval'], absolutes[0], np.amin(df_naive['peak_score']), color='#8C9196', alpha=0.25)
axes[0].set_ylim(0.25, 0.9)
axes[0].set_ylabel('Peak score ', fontsize=15)
axes[0].tick_params(labelsize=13)
axes[0].legend(loc='lower right', ncol=1, fontsize=15)

sns.lineplot(x='eval', y='naive_score', data=df_naive, ax=axes[1])
axes[1].set_yscale('log')
axes[1].axhline(absolutes[1], ls='--', linewidth=2, c='k', alpha=0.6)
axes[1].fill_between(df_naive['eval'], absolutes[1], np.amax(df_naive['naive_score']), color='#8C9196', alpha=0.25)
axes[1].set_ylim(np.amin(df_naive['naive_score']), np.amax(df_naive['naive_score']))
axes[1].set_ylabel('Naive score \n$( \$ \cdot (mol \  target)^{-1}$)', fontsize=15)
axes[1].tick_params(labelsize=13)

sns.lineplot(x='eval', y='spectral_overlap', data=df_naive, ax=axes[2])
axes[2].axhline(absolutes[2], ls='--', linewidth=2, c='k', alpha=0.6)
axes[2].fill_between(df_naive['eval'], absolutes[2], np.amax(df_naive['spectral_overlap']), color='#8C9196', alpha=0.25)
axes[2].set_ylim(0., 0.3)
axes[2].set_ylabel('Spectral \noverlap', fontsize=15)
axes[2].tick_params(labelsize=13)

sns.lineplot(x='eval', y='fluo_rate', data=df_naive, ax=axes[3])
axes[3].axhline(absolutes[3], ls='--', linewidth=2, c='k', alpha=0.6)
axes[3].fill_between(df_naive['eval'], absolutes[3], np.amin(df_naive['fluo_rate']), color='#8C9196', alpha=0.25)
axes[3].set_ylim(0., 0.6)
axes[3].set_ylabel('Fluorescence \nrate (ns$^{-1}$)', fontsize=15)
axes[3].tick_params(labelsize=13)
axes[3].set_xlabel('Number of evaluations', fontsize=15)

for ax in axes:
    ax.set_xlim(0, 500)

plt.tight_layout()
plt.savefig('Figure_S18.png', dpi=300)
plt.show()
