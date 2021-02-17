import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.widgets import Slider, Button, RadioButtons

HERE = os.path.abspath(os.path.dirname(__file__))
PROP_DIR = os.path.join(HERE, 'Properties')

# Create Map
cm = plt.get_cmap("viridis")

fp = pd.read_pickle(os.path.join(PROP_DIR, '20210216_full_props.pkl'))
fp['log_RS'] = np.log10(fp['route_score'])
fp['log_kR'] = np.log10(fp['fluo_rate_ns'])

fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')

thPk = 0.67
thRS = 4.48
thOvlp = 0.20
thRate = -0.80

subSpace = fp[(fp['fluo_peak_1'] > thPk) & (fp['log_RS'] < thRS) & (fp['overlap'] < thOvlp) & (fp['log_kR'] > thRate)]
rmSpace = fp[~(fp['fluo_peak_1'] > thPk) | ~(fp['log_RS'] < thRS) | ~(fp['overlap'] < thOvlp) | ~(fp['log_kR'] > thRate)]
maxPkScr = fp[fp['fluo_peak_1'] == max(fp['fluo_peak_1'])]
maxCost = fp[fp['route_score'] == min(fp['route_score'])]
maxOvlp = fp[fp['overlap'] == min(fp['overlap'])]
maxRate = fp[fp['fluo_rate_ns'] == max(fp['fluo_rate_ns'])]

coverage = len(subSpace) / len(fp)

p1 = ax3D.scatter3D(rmSpace['log_RS'], rmSpace['overlap'], rmSpace['log_kR'], c='grey', marker='o', alpha=0.1)
# p2 = ax3D.scatter3D(subSpace['log_RS'], subSpace['overlap'], subSpace['log_kR'], c=cm(subSpace['fluo_peak_1']), marker='o', alpha=1)
p2 = ax3D.scatter3D(subSpace['log_RS'], subSpace['overlap'], subSpace['log_kR'], c=cm(subSpace['fluo_peak_1']), marker='o', alpha=1)
max1 = ax3D.scatter3D(maxPkScr['log_RS'], maxPkScr['overlap'], maxPkScr['log_kR'], c='red', marker='o', alpha=1, label='Peak score')
min2 = ax3D.scatter3D(maxCost['log_RS'], maxCost['overlap'], maxCost['log_kR'], c='purple', marker='o', alpha=1, label='RouteScore')
min3 = ax3D.scatter3D(maxOvlp['log_RS'], maxOvlp['overlap'], maxOvlp['log_kR'], c='orange', marker='o', alpha=1, label='Spectral overlap')
max4 = ax3D.scatter3D(maxRate['log_RS'], maxRate['overlap'], maxRate['log_kR'], c='blue', marker='o', alpha=1, label='Fluorescence rate')

fig.colorbar(p2,
	         ax=ax3D,
	         label='Peak score',
	         orientation='horizontal',
	         fraction=0.1,
	         pad=0.05,
	         shrink=0.5,
	         aspect=15
	         )

ax3D.set_xlabel('log(RouteScore)')
ax3D.set_xlim(3.25, 8.25)
ax3D.set_ylabel('Spectral overlap')
ax3D.set_zlabel('log(Fluorescence rate)')
# ax3D.set_title(f'thPk={thPk}, thRS={thRS}, thOvlp={thOvlp}, thRate={thRate} \n{len(subSpace)} molecules = {round(coverage*100, 1)}% of space')
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
