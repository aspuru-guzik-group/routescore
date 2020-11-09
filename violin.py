import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('scientific')

HERE = os.path.abspath(os.path.dirname(__file__))
RSLT_DIR = os.path.join(HERE, 'Results')

base: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'Base.pkl'))
base_man: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'Base_man.pkl'))
b: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'Buch.pkl'))
b_man: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'Buch_man.pkl'))
s: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'SNAr.pkl'))
s_man: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'SNAr_man.pkl'))
bs: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'B-S.pkl'))
bs_man: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'B-S_man.pkl'))
sb: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'S-B.pkl'))
sb_man: pd.DataFrame = pd.read_pickle(os.path.join(RSLT_DIR, 'S-B_man.pkl'))

results = [base, base_man, s, s_man, b, b_man, bs, bs_man, sb, sb_man]
lens = [len(df) for df in results]
print(lens)

all_dfs = pd.DataFrame()
for df in results:
    all_dfs = all_dfs.append(df, ignore_index=True)

bottom = -0.5
top = 4.5

ax = sns.violinplot(x='Set',
                    y='log(RouteScore)',
                    data=all_dfs,
                    scale='width',
                    order=['Base', 'SNAr', 'Buchwald', 'B-S', 'S-B']
                    )
ax.set_ylim(bottom=bottom, top=top)
plt.savefig('violin_set.pdf')
plt.show()

bx = sns.violinplot(x='Set',
                    y='log(RouteScore)',
                    hue='Type',
                    data=all_dfs,
                    split=True,
                    scale='width',
                    order=['Base', 'SNAr', 'Buchwald', 'B-S', 'S-B']
                    )
bx.set_ylim(bottom=bottom, top=top)
plt.savefig('violin_set+type.pdf')
plt.show()

cx = sns.violinplot(x='Manual steps',
                    y='log(RouteScore)',
                    data=all_dfs,
                    scale='width',
                    )
cx.set_ylim(bottom=bottom, top=top)
plt.savefig('violin_mansteps.pdf')
plt.show()

dx = sns.violinplot(x='Manual',
                    y='log(RouteScore)',
                    data=all_dfs,
                    scale='width',
                    order=['None', 'Before', 'After', 'Both']
                    )
dx.set_ylim(bottom=bottom, top=top)
plt.savefig('violin_mantype.pdf')
plt.show()
