import os
from typing import List
import pandas as pd
import numpy as np
from routescore import Calculate, Properties, Analysis

from rdkit import Chem
from rdkit.Chem import Descriptors

calc = Calculate()
pr = Properties()
an = Analysis()

HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
TARGETS_FILE = os.path.join(TGT_DIR, 'targets_Base.pkl')


targets: pd.DataFrame = pd.read_pickle(TARGETS_FILE)
targets['Step details'] = ''
targets['Step details'] = targets['Step details'].astype('object')

num_targets = len(targets.index)

print('Calculating RouteScore(s)...')
for i in range(len(targets.index)):
    steps: List[dict] = []

    target = targets.at[i, 'pentamer']

    # step 1
    sm_list: List[dict] = [
                           {'smiles': targets.at[i, 'a'], 'eq': 3},
                           {'smiles': targets.at[i, 'b'], 'eq': 1}
                           ]
    product = targets.at[i, 'ab']
    step1_scale: float = 0.0001
    step1_yld: float = 1
    man_rxn = False
    step1: dict = calc.StepScore(
                                 sm_list,
                                 product,
                                 target,
                                 'Suzuki',
                                 step1_scale,
                                 step1_yld,
                                 man_rxn
                                 )
    steps.append(step1)

    # step 2
    sm_list: List[dict] = [
                           {'smiles': targets.at[i, 'ab'], 'eq': 3},
                           {'smiles': targets.at[i, 'c'], 'eq': 1}
                           ]
    step2_scale: float = step1_scale * step1_yld
    step2_yld: float = 1
    man_rxn = False
    step2: dict = calc.StepScore(
                                 sm_list,
                                 target,
                                 target,
                                 'Suzuki',
                                 step2_scale,
                                 step2_yld,
                                 man_rxn
                                 )
    steps.append(step2)

    n_target = step2_yld * step2_scale
    iso = n_target * Descriptors.MolWt(Chem.MolFromSmiles(target))
    # print(f'Isolated yield:    {iso} g')

    route_score, total_man = calc.RouteScore(steps, n_target)
    print(total_man)
    # print('RouteScore =', route_score, '\n')

    targets.at[i, 'Isolated'] = iso
    targets.at[i, 'RouteScore'] = route_score
    targets.at[i, 'log(RouteScore)'] = np.log10(route_score)
    targets.at[i, 'Step details'] = steps
    targets.at[i, 'Total manual steps'] = total_man

targets = pr.get_props(targets)

targets.to_pickle(TARGETS_FILE)

an.plotting(targets)
