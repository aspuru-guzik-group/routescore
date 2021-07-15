#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# load random runs
df_rand = pd.read_pickle('Optimization/runs/frac_found_random.pkl')

# load gryffin runs
df_gryf = pd.read_pickle('Optimization/runs/frac_found_gryffin.pkl')


# make plot

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

sns.lineplot(data=df_rand, x='iter', y='per',
             lw=3,
             label='Random Sampling',
             ax=ax)
sns.lineplot(data=df_gryf, x='iter', y='per',
             lw=3,
             label='Gryffin + Chimera',
             ax=ax)

ax.legend(loc='upper left', fontsize=15)
ax.set_xlabel('Number of evaluations', fontsize=15)
ax.set_ylabel('Acceptable\n molecules (%)', fontsize=15)
ax.set_xlim(0, 500)
ax.set_ylim(0., 41.)
ax.tick_params(labelsize=15)

plt.tight_layout()
plt.show()
