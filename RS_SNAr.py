import os, pickle
from typing import List
import pandas as pd
from routescore import General, Reaction_Templates, Calculate, Properties, Analysis

gen = General()
rxn = Reaction_Templates()
calc = Calculate()
pr = Properties()
an = Analysis()

HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
TARGETS_FILE = os.path.join(TGT_DIR, 'targets_SNAr.pkl')


# targets: pd.DataFrame = pd.read_pickle(TARGETS_FILE)
# targets['Step details'] = ''
# targets['Step details'] = targets['Step details'].astype('object')
targets: pd.DataFrame = gen.preProcess(TARGETS_FILE)

num_targets = len(targets.index)

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
                                               targets.at[i, 'F'],
                                               target,
                                               scale
                                               )

    steps.append(step2)

    # step 3
    scale = NextStep_scale
    step3, NextStep_scale = rxn.SNAr(
                                     targets.at[i, 'F'],
                                     targets.at[i, 'carbazole'],
                                     target,
                                     target,
                                     scale
                                     )

    steps.append(step3)

    final_scale = NextStep_scale

    targets = gen.Process(
                          targets,
                          i,
                          steps,
                          final_scale
                          )

targets = pr.get_props(targets)

targets.to_pickle(TARGETS_FILE)

an.plotting(targets)
