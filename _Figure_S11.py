#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

# distributions of the individual scores

# load the full properties
df_full = pickle.load(open('Properties/full_props.pkl', 'rb'))


# pairwise correlations of the naive score and SA, SC, SYBA, and RAscore-NN
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
axes = axes.flatten()

scores = ['sa_score', 'sc_score', 'syba_score', 'sr_nn_score']
names = ['SAscore', 'SCscore', 'SYBAscore', 'RAscore-NN']

for ix, (ax, score, name) in enumerate(zip(axes, scores, names)):

    sns.scatterplot(
            df_full[score],
            df_full['naive_score'],
            ax=ax,
        )
    ax.set_xlabel(name, fontsize=12)

    spear = spearmanr(df_full[score], df_full['naive_score'])
    pear = pearsonr(df_full[score], df_full['naive_score'])
    axes[ix].set_title(f'pearson = {round(pear[0], 2)}\nspearman = {round(spear[0], 2)}', fontsize=12)

axes[0].set_ylabel('Naive score $( \$ \cdot mol \ target^{-1})$', fontsize=12)

plt.tight_layout()
plt.savefig('Figure_S11.png', dpi=300)
plt.show()
