#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# load the raw data
df = pd.read_pickle('Properties/full_props.pkl')

# load the raw spectra
spec = pd.read_pickle('Properties/raw_spectra.pkl')

# helper functions
def overlap(absorb, emission):
    return np.sum(absorb * emission) / np.sqrt(np.sum(absorb * absorb) * np.sum(emission * emission))

#  function to calculate peak scores
def peak_scorer(roi, x_nm, spectra):
    # area in the roi
    roi_spectra = spectra[ x_nm<roi[1] ][ x_nm >= roi[0] ]
    roi_x_nm = x_nm[ x_nm < roi[1] ][ x_nm >= roi[0] ]
    roi_area = trapz(roi_spectra,roi_x_nm)
    tot_area = trapz(spectra, x_nm)
    return roi_area/tot_area

def get_spectra(spectra, smiles):
    d = list(filter(lambda mol: mol['smiles'] == smiles+'\n', spectra))[0]['spec']
    return d

min_peak_score = df[df['fluo_peak_1']==df['fluo_peak_1'].min()]
max_peak_score = df[df['fluo_peak_1']==df['fluo_peak_1'].max()]
min_overlap = df[df['overlap']==df['overlap'].min()]
max_overlap = df[df['overlap']==df['overlap'].max()]
min_peak_score['smiles'].tolist()[0]

min_peak_score_spec = get_spectra(spec, min_peak_score['smiles'].tolist()[0])
max_peak_score_spec = get_spectra(spec, max_peak_score['smiles'].tolist()[0])

min_overlap_spec = get_spectra(spec, min_overlap['smiles'].tolist()[0])
max_overlap_spec = get_spectra(spec, max_overlap['smiles'].tolist()[0])

# make plot

fluo = '#011627'
ext  = '#ff3366'
lw   = 3

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(10, 4))

# minimum peak score --------------------------------------------------------------
axes[0, 0].plot(min_peak_score_spec['x_nm'],
                min_peak_score_spec['ext_norm'],
                 label='extinction',
               lw=lw,
               c=ext)
axes[0, 0].plot(min_peak_score_spec['x_nm'],
                min_peak_score_spec['fluo_psd_norm'],
                label='fluorescence',
               lw=lw,
               c=fluo)
axes[0, 0].set_title(f'minimum peak score={round(min_peak_score["fluo_peak_1"].tolist()[0], 4)}')
axes[0, 0].axvspan(400, 460, alpha=0.2, color='#95B9DB')
axes[0, 0].axvline(400, color='#95B9DB')
axes[0, 0].axvline(460, color='#95B9DB')
axes[0, 0].set_xlim(200, 550.)

# maximum peak score --------------------------------------------------------------
axes[0, 1].plot(max_peak_score_spec['x_nm'],
                max_peak_score_spec['ext_norm'],
                lw=lw,
                c=ext)
axes[0, 1].plot(max_peak_score_spec['x_nm'],
                max_peak_score_spec['fluo_psd_norm'],
                lw=lw,
                c=fluo)
axes[0, 1].set_title(f'maximum peak score={round(max_peak_score["fluo_peak_1"].tolist()[0],3)}')
axes[0, 1].axvspan(400, 460, alpha=0.2, color='#95B9DB')
axes[0, 1].axvline(400, color='#95B9DB')
axes[0, 1].axvline(460, color='#95B9DB')
axes[0, 1].set_xlim(200, 550.)


# minimum spectral overlap --------------------------------------------------------
axes[1, 0].plot(min_overlap_spec['x_nm'],
                min_overlap_spec['ext_norm'],
                lw=lw,
                c=ext)
axes[1, 0].plot(min_overlap_spec['x_nm'],
                min_overlap_spec['fluo_psd_norm'],
                lw=lw,
                c=fluo)
axes[1, 0].set_title(f'minimum spectral overlap={round(min_overlap["overlap"].tolist()[0],3)}')
axes[1, 0].set_xlim(200., 800.)


# maximum spectral overlap --------------------------------------------------------
axes[1, 1].plot(max_overlap_spec['x_nm'],
                max_overlap_spec['ext_norm'],
                label='extinction',
                lw=lw,
                c=ext)
axes[1, 1].plot(max_overlap_spec['x_nm'],
                max_overlap_spec['fluo_psd_norm'],
                label='fluorescence',
                lw=lw,
               c=fluo)

axes[0, 0].legend(fontsize=12)
axes[1, 1].set_title(f'maximum spectral overlap={round(max_overlap["overlap"].tolist()[0],2)}')
axes[1, 1].set_xlim(300., 550.)


axes[0, 0].set_ylabel('Normalized spectra', fontsize=12)
axes[1, 0].set_ylabel('Normalized spectra', fontsize=12)

axes[1, 0].set_xlabel('Wavelength [nm]', fontsize=12)
axes[1, 1].set_xlabel('Wavelength [nm]', fontsize=12)



plt.tight_layout()
plt.show()
