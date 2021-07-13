#!/usr/bin/env python

import os, sys
import numpy as np
import pandas as pd
import pickle
import shutil

from gryffin import Gryffin

#=========================
# Define helper functions
#=========================

def full_measurement(a_ix, b_ix, c_ix, df):
    row = df.loc[(df['frag_a_ix']==a_ix) &
                 (df['frag_b_ix']==b_ix) &
                 (df['frag_c_ix']==c_ix)]
    # print(row)
    peak_score  = np.float(row['fluo_peak_1'])  # maximize
    route_score = np.float(row['route_score'])   # minimize
    spectral_overlap = np.float(row['overlap']) # minimize
    fluo_rate = np.float(row['fluo_rate_ns'])   # maximize

    return peak_score, route_score, spectral_overlap, fluo_rate

def load_prev_run(seed, type):
    params = []
    measurements = {
        'peak_score': [],
        'route_score': [],
        'spectral_overlap': [],
        'fluo_rate': [],
    }
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

def eval_merit(param):
    a_ix = int(param['A'].split('_')[-1])
    b_ix = int(param['B'].split('_')[-1])
    c_ix = int(param['C'].split('_')[-1])

    peak_score, route_score, spectral_overlap, fluo_rate = full_measurement(
        a_ix, b_ix, c_ix, df,
    )
    param['peak_score'] = peak_score
    param['route_score'] = route_score
    param['overlap'] = spectral_overlap
    param['fluo_rate'] = fluo_rate

    return param


#----------------
# Global settings
#----------------

ACQUISITION_OPTIMIZER = "adam"
BUDGET = 500                  # experimental budget for each run
REPEATS = 20                  # number of independent runs on this node
NODE_IX = int(sys.argv[1])    # 0, 1, 2
RANDOM_SEEDS = np.arange(REPEATS)+(NODE_IX*REPEATS)
SAMPLING_STRATEGIES = np.array([1, -1])

#----------------------
# Load the lookup table
#----------------------
df = pd.read_csv('targets/20210511_full_props.csv')


# loop over the random seeds
for seed in RANDOM_SEEDS:

    if os.path.isfile(f'runs_w_rs/gryf_naive_{seed}.pkl') is True:
        with open(f'runs_w_rs/gryf_naive_{seed}.pkl', 'rb') as content:
            past_data = pickle.load(content)
        if len(past_data) == 500:
            continue

    #========
    # Config
    #========
    A_frags = {f"frag_{i}":None for i in df.loc[: , 'frag_a_ix'].unique()}
    B_frags = {f"frag_{i}":None for i in df.loc[: , 'frag_b_ix'].unique()}
    C_frags = {f"frag_{i}":None for i in df.loc[: , 'frag_c_ix'].unique()}


    config = {
        "general": {
            "num_cpus": 8,
            "auto_desc_gen": False,
            "batches": 1,
            "sampling_strategies": 1,
            "boosted":  False,
            "caching": True,
            "random_seed": seed,
            "acquisition_optimizer": ACQUISITION_OPTIMIZER,
            "verbosity": 3
            },
        "parameters": [
            {"name": "A", "type": "categorical", "category_details": A_frags},
            {"name": "B", "type": "categorical", "category_details": B_frags},
            {"name": "C", "type": "categorical", "category_details": C_frags}
        ],
        # [0.67, 100000, 0.2, 0.15848931924611134]
        "objectives": [
            {"name": "peak_score", "goal": "max", "tolerance": 0.67, "absolute": True},
            {"name": "route_score", "goal": "min", "tolerance": 100000, "absolute": True},
            {"name": "overlap", "goal": "min", "tolerance": 0.2, "absolute": True},
            {"name": "fluo_rate", "goal": "max", "tolerance": 0.15848931924611134, "absolute": True}
        ]
    }

    # intialize gryffin
    gryffin = Gryffin(config_dict=config)

    observations = []
    print('====================================')
    print(f'Beginning run with seed {seed}')
    print('====================================')

    for iter in range(BUDGET):

        print('------------------------------------')
        print(f' RUN {seed} -- ITERATION : {iter}')
        print('------------------------------------')

        # select alternating sampling strategy
        select_ix = iter % len(SAMPLING_STRATEGIES)
        sampling_strategy = SAMPLING_STRATEGIES[select_ix]

        # query for new parameters
        params = gryffin.recommend(
            observations=observations, sampling_strategies=[sampling_strategy],
        )

        # select the single set of params we created
        param = params[0]

        # evaluated the proposed parameters
        observation = eval_merit(param)

        # append the new observation to the list
        observations.append(observation)

        # save the observations to disk with pickle
        pickle.dump(
            observations,
            open(f'runs_w_rs/gryf_naive_{seed}.pkl', 'wb')
        )
