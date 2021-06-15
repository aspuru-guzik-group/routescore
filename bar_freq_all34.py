import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import string
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display

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


dataset = 'w_rs'
w_wo = 'with' if dataset == 'w_rs' else 'without'
print(f'Frequency for satisfactory molecules {w_wo} RS')

opt_results = pd.read_csv(f'after_500_{dataset}.csv')
sat_df = pd.DataFrame()
sat_df = pd.read_csv('tolerable_34.csv')
sat_df.reset_index(inplace=True)
sat_df = sat_df.rename(columns={'index': 'mol_id'})
if dataset == 'w_rs':
    sat_df.mol_id += 1
elif dataset == 'rm_rs':
    sat_df.mol_id = list(string.ascii_uppercase)[:20]
else:
    print('Wrong dataset.')

for mol in opt_results.mol:
    if mol not in sat_df.smiles.tolist():
        img = Draw.MolToImage(Chem.MolFromSmiles(mol))
        display(img)

opt_mols = opt_results.mol.tolist()
mol_cts = [opt_mols.count(mol) for mol in sat_df.smiles]
if sum(mol_cts) != len(opt_results):
    print(f'ATTN: Counts != {len(opt_results)}')
    print(f'Count = {sum(mol_cts)}')
sat_df['freqs'] = [100*(ct/len(opt_results)) for ct in mol_cts]

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
# ax = sns.barplot(x='mol_id', y='freqs', ax=ax, data=sat_df, palette=np.array(pal[::-1])[rank])
# ax = sns.barplot(x='mol_id', y='freqs', ax=ax, data=sat_df, palette='crest')
ax.set(xlim=(-0.5, 33.5))
plt.xlabel('Satisfactory molecules')
plt.ylabel('Frequency (%)')
# plt.xticks(ticks=sat_df['mol_id'], labels=sat_df['mol_id'][sat_df.freqs != 0].tolist())
plt.tight_layout()
plt.savefig(f'all34_freq_{dataset}.pdf')
plt.show()

# plt.bar(sat_df.mol_id, sat_df.freqs, color='crest')
# plt.xlabel('Top 20 molecules with RouteScore')
# plt.ylabel('Frequency (%)')
# plt.tight_layout()
# plt.savefig(f'top_20_freq_{dataset}.pdf')
# plt.show()
