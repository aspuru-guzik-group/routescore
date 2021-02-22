import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('scientific')

fpath = '/Users/mseifrid/Dropbox (Aspuru-Guzik Lab)/Matter Lab/Projects/Subway synthesis/preproduction/subway_maps/Properties/spectra_dict.pkl'

with open(fpath, 'rb') as file:
	props_dict = pickle.load(file)

pmol = 'Fc1ccc(-c2ccsc2-c2cc(-c3sccc3-c3ccc(F)c4ccccc34)c3ccccc3c2)c2ccccc12'
pmol_df = props_dict[pmol]
# print(pmol_df.columns)

abs_norm = pmol_df['extinct'] / max(pmol_df['extinct'])
em_norm = pmol_df['fluo'] / max(pmol_df['fluo'])
nm = pmol_df['x_nm']

plt.plot(nm, abs_norm, label='Normalized absorbance')
plt.plot(nm, em_norm, label='Normalized emission')
plt.xlim(200, 800)
plt.xlabel('Wavelength (nm)')
plt.legend()
plt.show()