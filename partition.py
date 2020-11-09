import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
IPythonConsole.ipython_useSVG = True

HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
targets = pd.read_csv(os.path.join(TGT_DIR, 'full targets', 'targets_MASTER.csv'))

f_patt = Chem.MolFromSmarts('cF')
b_patt = Chem.MolFromSmiles('c12n(c3nccnc3)ccc1cccc2')
s_patt = Chem.MolFromSmiles('Cn(c1c2cccc1)c3c2cccc3')
cf = Chem.MolFromSmiles('CF')
pyr = Chem.MolFromSmiles('Nc1nccnc1')
nh = Chem.MolFromSmiles('N[H]')
nboc = Chem.MolFromSmiles('NC(OC(C)(C)C)=O')

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
    # if cond_s and cond_b and cond_f:
    #     g = g.append(targets.iloc[i], ignore_index=True)
    # if cond_s and (not cond_b) and cond_f:
    #     g = g.append(targets.iloc[i], ignore_index=True)
    if (not cond_s) and (not cond_b):
        n = n.append(targets.iloc[i], ignore_index=True)
    # if (not cond_s) and (not cond_b) and (not cond_f):
    #     n = n.append(targets.iloc[i], ignore_index=True)
    if cond_s and (not cond_b):
        s = s.append(targets.iloc[i], ignore_index=True)
    if (not cond_s) and cond_b:
        b = b.append(targets.iloc[i], ignore_index=True)
    # if (not cond_s) and cond_b:
    #     b = b.append(targets.iloc[i], ignore_index=True)
    if cond_s and cond_b:
        sb = sb.append(targets.iloc[i], ignore_index=True)
        bs = bs.append(targets.iloc[i], ignore_index=True)

print('\n––––––––––––––––––––––––––––––––––––––\n')
print('Targets:', len(targets))
print('n:', len(n))
print('b:', len(b))
print('s:', len(s))
print('sb:', len(sb))
print('g:', len(g))
print(f'{len(targets)} - ({len(n)} + {len(b)} + {len(s)} + {len(sb)} + {len(g)}) = {len(targets)-(len(n)+len(b)+len(s)+len(sb)+len(g))}')

print('\n––––––––––––––––––––––––––––––––––––––\n')

def abc_replace(df: pd.DataFrame, match: str):
    base_mol: dict = n[n['pentamer'] == df.at[i, match]].to_dict(orient='records')[0]
    df.at[i, 'a'] = base_mol['a']
    df.at[i, 'ab'] = base_mol['ab']
    df.at[i, 'b'] = base_mol['b']
    df.at[i, 'c'] = base_mol['c']

for i in range(len(s)):
    mol = Chem.MolFromSmiles(s.at[i, 'pentamer'])

    s.at[i, 'F'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, s_patt, cf, replaceAll=True)[0])
    s.at[i, 'carbazole'] = 'c1ccc2c(c1)[nH]c1ccccc12'

    abc_replace(s, 'F')

print('s:', len(s))
print('\n––––––––––––––––––––––––––––––––––––––\n')

for i in range(len(b)):
    mol = Chem.MolFromSmiles(b.at[i, 'pentamer'])

    b.at[i, 'N-Boc'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, pyr, nboc, replaceAll=True)[0])
    b.at[i, 'N-H'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, pyr, nh, replaceAll=True)[0])
    b.at[i, 'halide'] = 'Brc1nccnc1'

    abc_replace(b, 'N-Boc')

print('b:', len(b))
print('\n––––––––––––––––––––––––––––––––––––––\n')


for i in range(len(bs)):
    mol = Chem.MolFromSmiles(sb.at[i, 'pentamer'])
    buch_mol = Chem.rdmolops.ReplaceSubstructs(mol, s_patt, cf, replaceAll=True)[0]

    bs.at[i, 'N-Boc'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(buch_mol, pyr, nboc, replaceAll=True)[0])
    bs.at[i, 'N-H'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(buch_mol, pyr, nh, replaceAll=True)[0])
    bs.at[i, 'halide'] = 'Brc1nccnc1'

    bs.at[i, 'F'] = Chem.MolToSmiles(buch_mol)
    bs.at[i, 'carbazole'] = 'c1ccc2c(c1)[nH]c1ccccc12'

    abc_replace(bs, 'N-Boc')

print('bs:', len(bs))
print('\n––––––––––––––––––––––––––––––––––––––\n')

for i in range(len(sb)):
    mol = Chem.MolFromSmiles(sb.at[i, 'pentamer'])
    snar_mol = Chem.rdmolops.ReplaceSubstructs(mol, pyr, nboc, replaceAll=True)[0]

    sb.at[i, 'F'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(snar_mol, s_patt, cf, replaceAll=True)[0])
    sb.at[i, 'carbazole'] = 'c1ccc2c(c1)[nH]c1ccccc12'

    sb.at[i, 'N-Boc'] = Chem.MolToSmiles(snar_mol)
    sb.at[i, 'N-H'] = Chem.MolToSmiles(Chem.rdmolops.ReplaceSubstructs(mol, pyr, nh, replaceAll=True)[0])
    sb.at[i, 'halide'] = 'Brc1nccnc1'

    abc_replace(sb, 'F')

print('sb:', len(bs))
print('\n––––––––––––––––––––––––––––––––––––––\n')

n.to_csv(os.path.join(TGT_DIR, 'targets_Base_MASTER.csv'), index=False)
n.to_pickle(os.path.join(TGT_DIR, 'targets_Base.pkl'))
s.to_csv(os.path.join(TGT_DIR, 'targets_SNAr_MASTER.csv'), index=False)
s.to_pickle(os.path.join(TGT_DIR, 'targets_SNAr.pkl'))
b.to_csv(os.path.join(TGT_DIR, 'targets_Buch_MASTER.csv'), index=False)
b.to_pickle(os.path.join(TGT_DIR, 'targets_Buch.pkl'))
bs.to_csv(os.path.join(TGT_DIR, 'targets_B-S_MASTER.csv'), index=False)
bs.to_pickle(os.path.join(TGT_DIR, 'targets_B-S.pkl'))
sb.to_csv(os.path.join(TGT_DIR, 'targets_S-B_MASTER.csv'), index=False)
sb.to_pickle(os.path.join(TGT_DIR, 'targets_S-B.pkl'))
