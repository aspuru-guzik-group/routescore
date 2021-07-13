import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

# Custom style
mpl.style.use('scientific')


def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    # normalized = (values - min(values)) / (max(values) - min(values))
    normalized = (values - 0) / (100 - 0)
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


# Load optimization results
opt_results = pickle.load(open('./Optimization/top/all_best_mols_w_rs.pkl', 'rb'))
opt_mols = [opt['smiles'] for opt in opt_results]
# Load dataframe of satisfactory 34 (1%) molecules
sat_df = pd.read_csv('./Optimization/top/tolerable_34_w_rs.csv')
sat_df.reset_index(inplace=True)
sat_df = sat_df.rename(columns={'index': 'mol_id'})
sat_df.mol_id += 1

# Count how often each molecule is chosen by the optimization
mol_cts = [opt_mols.count(mol) for mol in sat_df.smiles]
if sum(mol_cts) != len(opt_results):
    print(f'ATTN: Counts != {len(opt_results)}')
    print(f'Count = {sum(mol_cts)}')
# Check if there are optimization results not in the list of satisfactory molecules
for smi in opt_mols:
    if smi not in sat_df.smiles.tolist():
        Draw.MolToImage(Chem.MolFromSmiles(smi))
# Calculate frequency of identifying molecule as top target
sat_df['freqs'] = [100*(ct/len(opt_results)) for ct in mol_cts]

# Make figure
dims = (13, 4)
fig, ax = plt.subplots(figsize=dims)
pal = sns.color_palette('crest', len(sat_df))
rank = sat_df.freqs.argsort().argsort()
print(sat_df[sat_df.freqs != 0])
ax = sns.barplot(x='mol_id',
                 y='freqs',
                 ax=ax,
                 data=sat_df,
                 palette=colors_from_values(sat_df.freqs, 'crest_r'))
ax.set(xlim=(-0.5, 33.5))
plt.xlabel('Satisfactory molecules')
plt.ylabel('Frequency (%)')
plt.tight_layout()
plt.savefig('Figure_8.pdf')
plt.show()
