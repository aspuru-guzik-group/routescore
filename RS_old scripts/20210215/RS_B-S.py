import os
from typing import List
import pandas as pd
# import numpy as np
from routescore import Reaction_Templates, Calculate, Properties, Analysis
# from rdkit import Chem
# from rdkit.Chem import Descriptors

rxn = Reaction_Templates()
calc = Calculate()
pr = Properties()
an = Analysis()

HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
TARGETS_FILE = os.path.join(TGT_DIR, 'targets_B-S.pkl')

targets: pd.DataFrame = pd.read_pickle(TARGETS_FILE)
targets['Step details'] = ''
targets['Step details'] = targets['Step details'].astype('object')

num_targets = len(targets.index)

print('Calculating RouteScore(s)...')
for i in range(len(targets.index)):
    steps: List[dict] = []

    target = targets.at[i, 'pentamer']

    # step 1
    """
    sm_list: List[dict] = [{'smiles': targets.at[i, 'a'], 'eq': 3},
                           {'smiles': targets.at[i, 'b'], 'eq': 1}]
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
    """
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
    """
    sm_list: List[dict] = [
                           {'smiles': targets.at[i, 'ab'], 'eq': 3},    # This should be checked...
                           {'smiles': targets.at[i, 'c'], 'eq': 1}
                           ]
    product = targets.at[i, 'N-Boc']
    step2_scale: float = step1_scale * step1_yld / sm_list[0]['eq'] # Here we are dividing by eq of 'ab' because it's assumed that it is the limiting reagent since it's the only synthesized compound.
    step2_yld: float = 1
    man_rxn = False
    step2: dict = calc.StepScore(
                                 sm_list,
                                 product,
                                 target,
                                 'Suzuki',
                                 step2_scale,
                                 step2_yld,
                                 man_rxn
                                 )
    """
    scale = NextStep_scale
    step2, NextStep_scale = rxn.pentamerSuzuki(
                                               targets.at[i, 'ab'],
                                               targets.at[i, 'c'],
                                               targets.at[i, 'N-Boc'],
                                               target,
                                               scale
                                               )
    
    steps.append(step2)

    # step 3
    """
    sm_list: List[dict] = [
                           {'smiles': targets.at[i, 'N-Boc'], 'eq': 1}
                           ]
    product = targets.at[i, 'N-H']
    step3_scale: float = step2_scale * step2_yld / sm_list[0]['eq']
    step3_yld: float = 1
    man_rxn = True
    step3: dict = calc.StepScore(
                                 sm_list,
                                 product,
                                 target,
                                 'Buchwald_deprotection',
                                 step3_scale,
                                 step3_yld,
                                 man_rxn
                                 )
    """
    scale = NextStep_scale
    step3, NextStep_scale = rxn.deBoc(
                                      targets.at[i, 'N-Boc'],
                                      targets.at[i, 'N-H'],
                                      target,
                                      scale
                                      )
    
    steps.append(step3)

    # step 4
    """
    sm_list: List[dict] = [
                           {'smiles': targets.at[i, 'N-H'], 'eq': 1},
                           {'smiles': targets.at[i, 'halide'], 'eq': 3}
                           ]
    product = targets.at[i, 'F']
    step4_scale: float = step3_scale * step3_yld / sm_list[0]['eq']
    step4_yld: float = 1
    man_rxn = True
    step4: dict = calc.StepScore(
                                 sm_list,
                                 product,
                                 target,
                                 'Buchwald',
                                 step4_scale,
                                 step4_yld,
                                 man_rxn
                                 )
    """
    scale = NextStep_scale
    step4, NextStep_scale = rxn.BHA(
                                    targets.at[i, 'N-H'],
                                    targets.at[i, 'halide'],
                                    targets.at[i, 'F'],
                                    target,
                                    scale
                                    )
    
    steps.append(step4)

    # step 5
    """
    sm_list: List[dict] = [
                           {'smiles': targets.at[i, 'F'], 'eq': 1},
                           {'smiles': targets.at[i, 'carbazole'], 'eq': 2}
                           ]
    step5_scale: float = step4_scale * step4_yld / sm_list[0]['eq']
    step5_yld: float = 1
    man_rxn = True
    step5: dict = calc.StepScore(
                                 sm_list,
                                 target,
                                 target,
                                 'SNAr',
                                 step5_scale,
                                 step5_yld,
                                 man_rxn
                                 )
    """
    scale = NextStep_scale
    step5, NextStep_scale = rxn.SNAr(
                                     targets.at[i, 'F'],
                                     targets.at[i, 'carbazole'],
                                     target,
                                     target,
                                     scale
                                     )
    
    steps.append(step5)
    """
    final_scale = step5_yld * step5_scale
    """
    final_scale = NextStep_scale
    

    targets = calc.Process(
                           targets,
                           i,
                           steps,
                           final_scale
                           )

targets = pr.get_props(targets)

targets.to_pickle(TARGETS_FILE)

an.plotting(targets)
