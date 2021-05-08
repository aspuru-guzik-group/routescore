import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
PROP_DIR = os.path.join(HERE, 'Properties')

# Create colormap
cm = plt.get_cmap("viridis")

# Load dataframe with all properties
fp = pd.read_pickle(os.path.join(PROP_DIR, 'full_props.pkl'))
fp['log_RS'] = np.log10(fp['route_score'])
fp['log_kR'] = np.log10(fp['fluo_rate_ns'])

# Define thresholds
thPk = 0.67
thRS = 4.44
thOvlp = 0.20
thRate = -0.80

# Filter dataframes based on thresholds
subSpace = fp[(fp['fluo_peak_1'] > thPk) & (fp['log_RS'] < thRS) & (fp['overlap'] < thOvlp) & (fp['log_kR'] > thRate)]
rmSpace = fp[~(fp['fluo_peak_1'] > thPk) | ~(fp['log_RS'] < thRS) | ~(fp['overlap'] < thOvlp) | ~(fp['log_kR'] > thRate)]

# Calculate % tolerable space
coverage = len(subSpace) / len(fp)
print(f'thPk={thPk}, thRS={thRS}, thOvlp={thOvlp}, thRate={thRate} \n{len(subSpace)} molecules = {round(coverage*100, 1)}% of space')

# Best entries for each property
maxPkScr = fp[fp['fluo_peak_1'] == max(fp['fluo_peak_1'])]
minCost = fp[fp['route_score'] == min(fp['route_score'])]
minOvlp = fp[fp['overlap'] == min(fp['overlap'])]
maxRate = fp[fp['log_kR'] == max(fp['log_kR'])]

# Create figure
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')

# 3D scatter plots
p1 = ax3D.scatter3D(rmSpace['log_RS'], rmSpace['overlap'], rmSpace['log_kR'], c='grey', marker='o', alpha=0.1)
p2 = ax3D.scatter3D(subSpace['log_RS'], subSpace['overlap'], subSpace['log_kR'], c=cm(subSpace['fluo_peak_1']), marker='o', alpha=1)
max1 = ax3D.scatter3D(maxPkScr['log_RS'], maxPkScr['overlap'], maxPkScr['log_kR'], c='red', marker='o', alpha=1, label='Peak score')
min2 = ax3D.scatter3D(minCost['log_RS'], minCost['overlap'], minCost['log_kR'], c='purple', marker='o', alpha=1, label='RouteScore')
min3 = ax3D.scatter3D(minOvlp['log_RS'], minOvlp['overlap'], minOvlp['log_kR'], c='orange', marker='o', alpha=1, label='Spectral overlap')
max4 = ax3D.scatter3D(maxRate['log_RS'], maxRate['overlap'], maxRate['log_kR'], c='blue', marker='o', alpha=1, label='Fluorescence rate')

# Colorbar
fig.colorbar(p2,
             ax=ax3D,
             label='Peak score',
             orientation='horizontal',
             fraction=0.1,
             pad=0.05,
             shrink=0.5,
             aspect=15
             )

ax3D.set_xlabel('\nlog(RouteScore)\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$')
ax3D.set_xlim(3.25, 8.25)
ax3D.set_ylabel('Spectral overlap')
ax3D.set_zlabel('log(Fluorescence rate) ($ns^{-1}$)')
ax3D.legend(
            title='Best:',
            bbox_to_anchor=(0.5, -0.2),
            loc='upper center',
            ncol=2,
            frameon=False,
            framealpha=0,
            mode=None
            )
plt.show()
