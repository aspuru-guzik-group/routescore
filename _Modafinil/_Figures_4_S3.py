import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.text import TextPath

# Custom style
mpl.style.use('scientific')

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))

n_routes = 10

# Load route dataframes
routes = [
          pd.read_excel(f'output_Route_{i}.xlsx', sheet_name='Summary')
          for i in range(1, n_routes+1)
          ]

# log(RouteScore)
logRS = [route['log RS'][0] for route in routes]
# RouteScore
RS = [route['full RS'][0] for route in routes]
# Route numbers
route_num = list(range(1, n_routes+1))
# Average time component over the route
avg_time = [route['avg C_time'][0] for route in routes]
# Average monetary component over the route
avg_money = [route['avg C_money'][0] for route in routes]
# Average mass component over the route
avg_mass = [route['avg C_mass'][0] for route in routes]
# Number of steps
steps = [route['# steps'][0] for route in routes]
# Scale of the route (mols)
scale = [route['n_Target'][0] for route in routes]
# total time for the route
time = [route['total C_time'][0] for route in routes]
# total yield
yld = [route['Total yield'][0] for route in routes]

# Make summary dataframe
summary = {
           'Route': route_num,
           'RouteScore': RS,
           'log(RouteScore)': logRS,
           'Avg. time cost per step (h)': avg_time,
           'Avg. monetary cost per step ($)': avg_money,
           'Avg. materials cost per step (g)': avg_mass,
           'Number of steps': steps,
           'Scale (mol)': scale,
           'Total human time (h)': time,
           'Total yield (%)': yld
            }
summary_df = pd.DataFrame.from_dict(summary)
summary_df.to_csv('Modaf_route_summaries.csv', index=False)

# Create colormap for number of steps
cmap = mpl.colors.ListedColormap(['#5e3c99', '#b2abd2', '#fdb863', '#e66101'])
bounds = [1, 2, 3, 4, 5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create Figure 4
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

route_num_txt = [str(num)for num in route_num]
for y, t, rs, s, rt in zip(yld, time, logRS, steps, route_num_txt):
    color = np.array([s])
    path = TextPath((0,0), rt)
    size = 1600 if rt == '10' else 800
    xyz1 = ax1.scatter3D(y, t, rs, c=color, cmap=cmap, norm=norm, s=size, marker=path, alpha=1)
ax1.set_xlabel('Overall yield (%)')
ax1.set_ylabel('Total human time (h)')
ax1.set_zlabel('\nlog(RouteScore)\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$', rotation=270)

# Make colorbar
cbar = plt.colorbar(xyz1,
                    ax=ax1,
                    cmap=cmap,
                    boundaries=bounds,
                    ticks=bounds,
                    spacing='proportional',
                    label='Number of steps',
                    orientation='horizontal',
                    shrink=0.33,
                    aspect = 10,
                    pad=0.1
                    )
labels = np.linspace(min(steps), max(steps), len(set(steps))).astype(int)
loc = labels + 0.5
cbar.set_ticks(loc)
cbar.set_ticklabels(labels)
plt.show()

# Make plots for Figure S3
for component in summary_df.columns:
    if component in ('Avg. monetary cost per step ($)', 'Avg. materials cost per step (g)', 'Scale (mol)'):
        x_log = True
        plt_xscale = 'log'
    else:
        x_log = False
        plt_xscale = 'linear'
    for i in range(n_routes):
        path = TextPath((0,0), route_num_txt[i])
        size = 800 if route_num_txt[i] == '10' else 400
        plt.scatter(summary_df[component][i], summary_df['RouteScore'][i], s=size, marker=path)
    plt.yscale('log')
    plt.xscale(plt_xscale)
    plt.ylabel('RouteScore\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$')
    plt.xlabel(component)
    plt.savefig(os.path.join(HERE, 'Figure_S3', f'RS vs {component}.png'))
    plt.show()
