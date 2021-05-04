import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Custom style
mpl.style.use('scientific')

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
PROP_DIR = os.path.join(HERE, 'Properties')

# Create colormap
cm = plt.get_cmap("viridis")

# Load dataframe with all properties
fp = pd.read_pickle(os.path.join(PROP_DIR, 'full_props.pkl'))
fp['log_RS'] = np.log10(fp['route_score'])
fp['log_kR'] = np.log10(fp['fluo_rate_ns'])

# Create figure
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')

# 3D scatter plot
ax = ax3D.scatter3D(fp['log_RS'], fp['overlap'], fp['log_kR'], c=cm(fp['fluo_peak_1']), marker='o')

# Colorbar
fig.colorbar(ax,
             ax=ax3D,
             label='Peak score (a.u.)',
             pad=0.1,
             shrink=0.42,
             aspect=10,
             anchor=(0.0, 1.0),
             panchor=(1.0, 1.0),
             )

ax3D.set_xlabel('log(RouteScore) (a.u.)')
ax3D.set_xlim(3.75, 8.25)
ax3D.set_ylabel('Spectral overlap (a.u.)')
ax3D.set_zlabel('log(Fluorescence rate) ($ns^{-1}$)')
plt.show()
