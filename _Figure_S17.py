#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# distributions of the individual scores

# load the full properties
df_full = pickle.load(open('Properties/full_props.pkl', 'rb'))


scores = [
    'naive_score',
    'route_score',
    'sa_score',
    'sc_score',
    'syba_score',
    'sr_nn_score',
]

names = [
         'Naive score [$ mol$^{-1}$]',
         'RouteScore',
         'SAscore',
         'SCscore',
         'SYBAscore',
         'RAscore-NN',
    ]

fig, axes = plt.subplots(1, 6, figsize=(15, 3.5))
axes = axes.flatten()

for ix, (ax, score, name) in enumerate(zip(axes, scores, names)):
    if score == 'route_score':
        sns.distplot(
            #np.log10(df_full[score]),
            df_full[score],
            kde=False,
            ax=ax,
        )
    else:
        sns.distplot(
            df_full[score],
            kde=False,
            ax=ax,
        )

    if ix == 0:
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.set_xlabel('Naive score $(\$ \cdot mol^{-1})$', fontsize=12)
        ax.set_ylabel('Absolute frequency', fontsize=12)
    elif ix == 1:
        ax.set_xlabel('RouteScore\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$)', fontsize=12)
    else:
        ax.set_xlabel(name, fontsize=12)



plt.tight_layout()
plt.savefig('Figure_S17.png', dpi=300)
plt.show()
