import os
from typing import List
import pandas as pd
from routescore import General, Reaction_Templates, Calculate

gen = General()
rxn = Reaction_Templates()
calc = Calculate()

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
TARGETS_FILE = os.path.join(TGT_DIR, 'targets_Base.pkl')

# Load dataframe and add details columns
targets = gen.preProcess(TARGETS_FILE)

# Run RouteScore calculations
print('Calculating RouteScore(s)...')
for i in range(len(targets.index)):
    steps: List[dict] = []

    target = targets.at[i, 'pentamer']

    # step 1
    scale: float = 0.0001
    step1, NextStep_scale = rxn.wingSuzuki(
                                           targets.at[i, 'a'],
                                           targets.at[i, 'b'],
                                           targets.at[i, 'ab'],
                                           target,
                                           scale
                                           )
    steps.append(step1)

    # step 2
    scale = NextStep_scale
    step2, NextStep_scale = rxn.pentamerSuzuki(
                                               targets.at[i, 'ab'],
                                               targets.at[i, 'c'],
                                               target,
                                               target,
                                               scale
                                               )
    steps.append(step2)

    final_scale = NextStep_scale
    targets = gen.Process(
                          targets,
                          i,
                          steps,
                          final_scale
                          )

targets.to_pickle(TARGETS_FILE)
