import pickle
import matplotlib as mpl
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom style
mpl.style.use('scientific')

# Load data
props = pickle.load(open('./Properties/full_props.pkl', 'rb'))


plt.scatter(np.log10(props['route_score']), np.log10(props['naive_score']))
plt.xlabel('log(RouteScore)\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$)')
plt.ylabel('log(Naive chemical cost)\n$(\$ \cdot (mol \  target)^{-1}$)')
plt.tight_layout()
plt.savefig('Figure_S9a.pdf')
plt.show()

# Correlation for low RouteScore molecules
cheap = props[np.log10(props['route_score']) < 6]
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(cheap.route_score, cheap.naive_score)

plt.scatter(np.log10(cheap['route_score']), np.log10(cheap['naive_score']))
plt.xlabel('log(RouteScore)\n$(h \cdot \$ \cdot g \cdot (mol \  target)^{-1}$)')
plt.ylabel('log(Naive chemical cost)\n$(\$ \cdot (mol \  target)^{-1}$)')
plt.text(5.3, 5.5,
         f'$R^{2}$ = {round(r_value,2)}',
         horizontalalignment='right',
         verticalalignment='bottom')
plt.tight_layout()
plt.savefig('Figure_S9b.pdf')
plt.show()
