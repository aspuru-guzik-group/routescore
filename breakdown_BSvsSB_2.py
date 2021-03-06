import os
import shutil
from typing import List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
BRKDWN_DIR = os.path.join(HERE, 'BSvsSB_breakdown')

bs = pd.read_pickle(os.path.join(TGT_DIR, 'targets_B-S.pkl'))
sb = pd.read_pickle(os.path.join(TGT_DIR, 'targets_S-B.pkl'))

# Deleting previous images and directories
shutil.rmtree(BRKDWN_DIR)
# Creating new directory tree
os.mkdir(BRKDWN_DIR)
for step_detail in ['time', 'money', 'materials']:
    for rxn in ['Buchwald_deprotection', 'Buchwald', 'SNAr']:
        os.makedirs(os.path.join(BRKDWN_DIR, step_detail, rxn))


def extract_details(df):
    """Extract step details for last 3 steps (deBoc, BHA and SNAr)."""
    df_RSinfo = df[['pentamer', 'Step details', 'RouteScore details',
                    'Isolated', 'RouteScore', 'log(RouteScore)']]

    last3_rxns = ['Buchwald_deprotection', 'Buchwald', 'SNAr']
    for rxn in last3_rxns:
        df_RSinfo[rxn] = [next(step for step in row[-3:] if step['reaction'] == rxn)
                          for row in df['Step details']]

    for key in df_RSinfo['RouteScore details'][0].keys():
        # df_RSinfo[key] = [[row[key] for key in row.keys()] for row in df['RouteScore details']]
        df_RSinfo[key] = [row[key] for row in df['RouteScore details']]

    return df_RSinfo


def uniqueVals(df: pd.DataFrame, rxn: str, component: str) -> List[float]:
    """Extract list of unique values in component of reaction StepScore."""
    vals = [mol[component] for mol in df[rxn]]
    print(f'Unique values in {rxn} {component}: {len(set(vals))}\n')
    print(f'{rxn} {component}: {set(vals)}\n')
    return list(set(vals))


def Vals2Mols(df: pd.DataFrame, df_name: str, rxn: str, component: str, vals_list: List[float]) -> None:
    """Extract SMILES that correspond to val in component of reaction StepScore."""
    for val in vals_list:
        smiles_list: List[str] = []
        for i in range(len(df)):
            if df[rxn][i][component] == val:
                smiles_list.append(df['pentamer'][i])

        mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        # gen.draw_mols(smiles_list)

        img = Draw.MolsToGridImage(mols_list, molsPerRow=5, subImgSize=(500, 500), useSVG=False)
        img.save(os.path.join(BRKDWN_DIR,
                              component,
                              rxn,
                              f'{df_name}_{rxn}_{component}_{val}_mols.png'))


bs_RSinfo = extract_details(bs)
sb_RSinfo = extract_details(sb)

for df, name in zip([bs_RSinfo, sb_RSinfo], ['bs', 'sb']):
    for step_detail in ['time', 'money', 'materials']:
        for rxn in ['Buchwald_deprotection', 'Buchwald', 'SNAr']:
            Vals2Mols(df,
                      name,
                      rxn,
                      step_detail,
                      uniqueVals(df, rxn, step_detail))
