#!/usr/bin/env python

import numpy as np
import pandas as pd

import itertools
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns


# load the data
target_df = pd.read_pickle('Properties/full_props.pkl')


# make correlations plot
data_types = ['fluo_peak_1', 'route_score', 'overlap', 'fluo_rate_ns']
labels     = {'fluo_peak_1': 'Peak score',
              'route_score': 'RouteScore\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$',
              'overlap': 'Spectral overlap',
              'fluo_rate_ns': 'Fluorescence \nrate (ns$^{-1}$)'}
combs = list(itertools.combinations(data_types, 2))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
axes = axes.flatten()
pal = sns.color_palette('rocket')

for ix, comb in enumerate(combs):

    sns.scatterplot(target_df[comb[0]], target_df[comb[1]],
                    c=pal[ix], alpha=0.8, ax=axes[ix])

    spear = spearmanr(target_df[comb[0]], target_df[comb[1]])
    pear  = pearsonr(target_df[comb[0]], target_df[comb[1]])

    axes[ix].set_title(f'pearson = {round(pear[0], 2)}\nspearman = {round(spear[0], 2)}', fontsize=10)

    axes[ix].set_xlabel(labels[comb[0]], fontsize=12)
    axes[ix].set_ylabel(labels[comb[1]], fontsize=12)

plt.ticklabel_format(useOffset=False)

plt.tight_layout()
plt.show()
