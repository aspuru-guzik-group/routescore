#!/usr/bin/env python

import os, sys
import pickle 

import numpy as np
import pandas as pd

from category_writer import CategoryWriter
from gryffin import Gryffin
from merit_calculator import MeritCalculator


BUDGET      = 500
BATCH_SIZE = 2
CONFIG_FILE = 'config.json'
OBJECTIVE_TYPE = 'all'
seed = int(sys.argv[1])

TYPE = 'naive'

scratch_dir = f'./.scratch_dir_{seed}'
db_path     = f'./SearchProgress_{seed}'

#==========================================================================

df = pd.read_pickle('targets/20210308_full_props.pkl')

# write categories
category_writer = CategoryWriter()
category_writer.write_categories(home_dir = './', with_descriptors = False)


# initialize gryffin
gryffin = Gryffin(
        CONFIG_FILE, 
        random_seed=seed,
        scratch_dir=scratch_dir, 
        db_path=db_path
    )

#==========================================================================
# helper functions

def full_measurement(a_ix, b_ix, c_ix, df):
	row = df.loc[(df['frag_a_ix']==a_ix) & \
				 (df['frag_b_ix']==b_ix) & \
				 (df['frag_c_ix']==c_ix)]
	peak_score  = np.float(row['fluo_peak_1'])  # maximize
	route_score = np.float(row['route_score'])   # minimize
	spectral_overlap = np.float(row['overlap']) # minimize
	fluo_rate = np.float(row['fluo_rate_ns'])   # maximize

	return peak_score, route_score, spectral_overlap, fluo_rate

def load_prev_run(seed, type):
	params = []
	measurements = {'peak_score': [], 'route_score': [],
					'spectral_overlap': [], 'fluo_rate': []}
	if os.path.exists(f'runs/gryf_{type}_all_{seed}.pkl'):
		res = pickle.load(open(f'runs/gryf_{type}_all_{seed}.pkl', 'rb'))
		# populate the lists with previous results
		for ix, r in enumerate(res):
			p = {k: r[k] for k in list(r)[:3]} # first 3 k:v pairs --> frags
			params.append(p)
			measurements['peak_score'].append(r['peak_score'])
			measurements['route_score'].append(r['route_score'])
			measurements['spectral_overlap'].append(r['spectral_overlap'])
			measurements['fluo_rate'].append(r['fluo_rate'])
	else:
		# begin with empty params and measurements
		pass
	return params, measurements

#==========================================================================

#initialize the MeritCalculator
merit_calculator = MeritCalculator()
absolutes = merit_calculator.absolutes

params, measurements = load_prev_run(seed, TYPE)

observations = []
peak_scores = []
log_rs = []
overlaps = []
fluo_rates = []


# main loop
evaluations = 0

while evaluations < BUDGET:

	samples = gryffin.recommend(observations = observations)
	new_observations  = []
	for sample in samples:

		a_ix = int(sample['frag_a'][0])
		b_ix = int(sample['frag_b'][0])
		c_ix = int(sample['frag_c'][0])

		param = {'frag_a' : [f'{a_ix}'],
				  'frag_b' : [f'{b_ix}'],
				  'frag_c' : [f'{c_ix}'],}
		params.append(param)
		peak_score, route_score, spectral_overlap, fluo_rate = full_measurement(a_ix, b_ix, c_ix, df)
		measurements['peak_score'].append(peak_score)
		measurements['route_score'].append(route_score)
		measurements['spectral_overlap'].append(spectral_overlap)
		measurements['fluo_rate'].append(fluo_rate)

	new_observations = merit_calculator.get_observations(params, measurements)

	observations = new_observations
	pickle.dump(observations, open(f'runs/gryf_{TYPE}_{OBJECTIVE_TYPE}_{seed}.pkl', 'wb'))
	evaluations += BATCH_SIZE
