#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# absolute tolerances for chimera
absolutes = np.array([0.67, 100000, 0.2, 0.15848931924611134])

# load gryffn runs with RouteScore as objective
df_w_rs = pd.read_pickle('Optimization/runs/gryffin_runs_w_rs.pkl')

# load gryffn runs with RouteScore as objective
df_wo_rs = pd.read_pickle('Optimization/runs/gryffin_runs_wo_rs.pkl')


# make the plot

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 10))

sns.lineplot(x='eval', y='peak_score', data=df_w_rs, ax=axes[0], label='RouteScore included')
sns.lineplot(x='eval', y='peak_score', data=df_wo_rs, ax=axes[0], color='#6D1A36', label='RouteScore excluded')
axes[0].axhline(absolutes[0], ls='--', linewidth=2, c='k', alpha=0.6)
axes[0].fill_between(df_w_rs['eval'], absolutes[0], np.amin(df_w_rs['peak_score']), color='#8C9196', alpha=0.25)
axes[0].set_ylim(0.25, 0.9)
axes[0].set_ylabel('Peak score ', fontsize=15)
axes[0].tick_params(labelsize=13)
axes[0].legend(loc='lower right', ncol=1, fontsize=15)

sns.lineplot(x='eval', y='route_score', data=df_w_rs, ax=axes[1])
sns.lineplot(x='eval', y='route_score', data=df_wo_rs, ax=axes[1], color='#6D1A36')
axes[1].set_yscale('log')
axes[1].axhline(absolutes[1], ls='--', linewidth=2, c='k', alpha=0.6)
axes[1].fill_between(df_w_rs['eval'], absolutes[1], np.amax(df_w_rs['route_score']), color='#8C9196', alpha=0.25)
axes[1].set_ylim(np.amin(df_w_rs['route_score']), np.amax(df_w_rs['route_score']))
axes[1].set_ylabel('RouteScore\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$)', fontsize=15)
axes[1].tick_params(labelsize=13)

sns.lineplot(x='eval', y='spectral_overlap', data=df_w_rs, ax=axes[2])
sns.lineplot(x='eval', y='spectral_overlap', data=df_wo_rs, ax=axes[2], color='#6D1A36')
axes[2].axhline(absolutes[2], ls='--', linewidth=2, c='k', alpha=0.6)
axes[2].fill_between(df_w_rs['eval'], absolutes[2], np.amax(df_w_rs['spectral_overlap']), color='#8C9196', alpha=0.25)
axes[2].set_ylim(0., 0.3)
axes[2].set_ylabel('Spectral \noverlap', fontsize=15)
axes[2].tick_params(labelsize=13)

sns.lineplot(x='eval', y='fluo_rate', data=df_w_rs, ax=axes[3])
sns.lineplot(x='eval', y='fluo_rate', data=df_wo_rs, ax=axes[3], color='#6D1A36')
axes[3].axhline(absolutes[3], ls='--', linewidth=2, c='k', alpha=0.6)
axes[3].fill_between(df_w_rs['eval'], absolutes[3], np.amin(df_w_rs['fluo_rate']), color='#8C9196', alpha=0.25)
axes[3].set_ylim(0., 0.6)
axes[3].set_ylabel('Fluorescence \nrate (ns$^{-1}$)', fontsize=15)
axes[3].tick_params(labelsize=13)
axes[3].set_xlabel('Number of evaluations', fontsize=15)

for ax in axes:
    ax.set_xlim(0, 500)

plt.tight_layout()
plt.show()
