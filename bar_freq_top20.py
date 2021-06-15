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


dataset = 'rm_rs'
w_wo = 'with' if dataset == 'w_rs' else 'without'
print(f'Frequency for top 20 {w_wo} RS')

opt_results = pd.read_csv(f'after_500_{dataset}.csv')
top20_df = pd.read_csv(f'top_20_data_{dataset}.csv')
top20_df.reset_index(inplace=True)
top20_df = top20_df.rename(columns={'index': 'mol_id'})
if dataset == 'w_rs':
    top20_df.mol_id += 1
elif dataset == 'rm_rs':
    top20_df.mol_id = list(string.ascii_uppercase)[:20]
else:
    print('Wrong dataset.')

for mol in opt_results.mol:
    if mol not in top20_df.smiles.tolist():
        img = Draw.MolToImage(Chem.MolFromSmiles(mol))
        display(img)

opt_mols = opt_results.mol.tolist()
mol_cts = [opt_mols.count(mol) for mol in top20_df.smiles]
if sum(mol_cts) != len(opt_results):
    print(f'ATTN: Counts != {len(opt_results)}')
    print(f'Count = {sum(mol_cts)}')
top20_df['freqs'] = [100*(ct/len(opt_results)) for ct in mol_cts]

dims = (10, 4)
fig, ax = plt.subplots(figsize=dims)

pal = sns.color_palette('crest', len(top20_df))
rank = top20_df.freqs.argsort().argsort()

print(top20_df[top20_df.freqs != 0])
ax = sns.barplot(x='mol_id',
                 y='freqs',
                 ax=ax,
                 data=top20_df,
                 palette=colors_from_values(top20_df.freqs, 'crest_r'))
# ax = sns.barplot(x='mol_id', y='freqs', ax=ax, data=top20_df, palette=np.array(pal[::-1])[rank])
# ax = sns.barplot(x='mol_id', y='freqs', ax=ax, data=top20_df, palette='crest')
ax.set(xlim=(-1, 20))
plt.xlabel(f'Top 20 molecules {w_wo} RouteScore')
plt.ylabel('Frequency (%)')
plt.tight_layout()
plt.savefig(f'top_20_freq_{dataset}.pdf')
plt.show()

# plt.bar(top20_df.mol_id, top20_df.freqs, color='crest')
# plt.xlabel('Top 20 molecules with RouteScore')
# plt.ylabel('Frequency (%)')
# plt.tight_layout()
# plt.savefig(f'top_20_freq_{dataset}.pdf')
# plt.show()
