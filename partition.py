import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
IPythonConsole.ipython_useSVG = True

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
targets = pd.read_csv(os.path.join(TGT_DIR, 'full targets', 'targets_MASTER.csv'))


def abc_replace(df: pd.DataFrame, match: str):
    base_mol: dict = n[n['pentamer'] == df.at[i, match]].to_dict(orient='records')[0]
    df.at[i, 'a'] = base_mol['a']
    df.at[i, 'ab'] = base_mol['ab']
    df.at[i, 'b'] = base_mol['b']
    df.at[i, 'c'] = base_mol['c']


# Define patterns for matching
f_patt = Chem.MolFromSmarts('cF')
b_patt = Chem.MolFromSmiles('c12n(c3nccnc3)ccc1cccc2')
s_patt = Chem.MolFromSmiles('Cn(c1c2cccc1)c3c2cccc3')

# Define substructures for replacement
cf = Chem.MolFromSmiles('CF')
pyr = Chem.MolFromSmiles('Nc1nccnc1')
nh = Chem.MolFromSmiles('N[H]')
nboc = Chem.MolFromSmiles('NC(OC(C)(C)C)=O')

# Define starting materials to add
carbazole = 'c1ccc2c(c1)[nH]c1ccccc12'
BHA_halide = 'Brc1nccnc1'

# Split 'targets' dataframe into different route types by substructure matching
bs = pd.DataFrame()
sb = pd.DataFrame()
s = pd.DataFrame()
b = pd.DataFrame()
n = pd.DataFrame()
g = pd.DataFrame()
for i in range(len(targets)):
    cond_s = Chem.MolFromSmiles(targets.at[i, 'pentamer']).HasSubstructMatch(s_patt)
    cond_b = Chem.MolFromSmiles(targets.at[i, 'pentamer']).HasSubstructMatch(b_patt)
    cond_f = Chem.MolFromSmiles(targets.at[i, 'pentamer']).HasSubstructMatch(f_patt)
    
    if (not cond_s) and (not cond_b):
        n = n.append(targets.iloc[i], ignore_index=True)
    
    if cond_s and (not cond_b):
        s = s.append(targets.iloc[i], ignore_index=True)
    
    if (not cond_s) and cond_b:
        b = b.append(targets.iloc[i], ignore_index=True)
    
    if cond_s and cond_b:
        sb = sb.append(targets.iloc[i], ignore_index=True)
        bs = bs.append(targets.iloc[i], ignore_index=True)

# Check that everything was split correctly
print('\n––––––––––––––––––––––––––––––––––––––\n')
print('Targets:', len(targets))
print('n:', len(n))
print('b:', len(b))
print('s:', len(s))
print('sb:', len(sb))
print('g:', len(g))
print(f'{len(targets)} - ({len(n)} + {len(b)} + {len(s)} + {len(sb)} + {len(g)}) = {len(targets)-(len(n)+len(b)+len(s)+len(sb)+len(g))}')

print('\n––––––––––––––––––––––––––––––––––––––\n')

# Add the corresponding intermediates for each route
for i in range(len(s)):
    mol = Chem.MolFromSmiles(s.at[i, 'pentamer'])

    s.at[i, 'F'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, s_patt, cf, replaceAll=True)[0])
    s.at[i, 'carbazole'] = carbazole

    abc_replace(s, 'F')

print('s:', len(s))
print('\n––––––––––––––––––––––––––––––––––––––\n')

for i in range(len(b)):
    mol = Chem.MolFromSmiles(b.at[i, 'pentamer'])

    b.at[i, 'N-Boc'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, pyr, nboc, replaceAll=True)[0])
    b.at[i, 'N-H'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, pyr, nh, replaceAll=True)[0])
    b.at[i, 'halide'] = BHA_halide

    abc_replace(b, 'N-Boc')

print('b:', len(b))
print('\n––––––––––––––––––––––––––––––––––––––\n')

for i in range(len(bs)):
    mol = Chem.MolFromSmiles(sb.at[i, 'pentamer'])
    buch_mol = Chem.rdmolops.ReplaceSubstructs(mol, s_patt, cf, replaceAll=True)[0]

    bs.at[i, 'N-Boc'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(buch_mol, pyr, nboc, replaceAll=True)[0])
    bs.at[i, 'N-H'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(buch_mol, pyr, nh, replaceAll=True)[0])
    bs.at[i, 'halide'] = BHA_halide

    bs.at[i, 'F'] = Chem.MolToSmiles(buch_mol)
    bs.at[i, 'carbazole'] = carbazole

    abc_replace(bs, 'N-Boc')

print('bs:', len(bs))
print('\n––––––––––––––––––––––––––––––––––––––\n')

for i in range(len(sb)):
    mol = Chem.MolFromSmiles(sb.at[i, 'pentamer'])
    snar_mol = Chem.rdmolops.ReplaceSubstructs(mol, pyr, nboc, replaceAll=True)[0]

    sb.at[i, 'F'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(snar_mol, s_patt, cf, replaceAll=True)[0])
    sb.at[i, 'carbazole'] = carbazole

    sb.at[i, 'N-Boc'] = Chem.MolToSmiles(snar_mol)
    sb.at[i, 'N-H'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, pyr, nh, replaceAll=True)[0])
    sb.at[i, 'halide'] = BHA_halide

    abc_replace(sb, 'F')

print('sb:', len(sb))
print('\n––––––––––––––––––––––––––––––––––––––\n')

# Save dataframes to file
route_dfs = [n, s, b, bs, sb]
names = ['Base', 'SNAr', 'Buch', 'B-S', 'S-B']
for df, name in zip(route_dfs, names):
    df.to_csv(os.path.join(TGT_DIR, f'target_{name}_BLANK.csv'))
    df.to_pickle(os.path.join(TGT_DIR, f'targets_{name}.pkl'))
