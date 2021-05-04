import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Custom style
plt.style.use('scientific')

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
RSLT_DIR = os.path.join(HERE, 'Results')

# Load RouteScore dataframes
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

# Show size of dataframes
results = [base, base_man, s, s_man, b, b_man, bs, bs_man, sb, sb_man]
lens = [len(df) for df in results]
print(lens)

# Combine all dataframes into one
all_dfs = pd.DataFrame()
for df in results:
    all_dfs = all_dfs.append(df, ignore_index=True)

# Set y limits
bottom = 4
top = 8.5

ax = sns.violinplot(x='Route type',
                    y='log(RouteScore)',
                    data=all_dfs,
                    scale='width',
                    order=['iSMC auto.', 'iSMC man.', 'SNAr', 'BHA', 'B-S', 'S-B']
                    )
ax.set_ylim(bottom=bottom, top=top)
ax.set_xticklabels(['iSMC\nauto.', 'iSMC\nman.', 'SNAr', 'BHA', 'B-S', 'S-B'])
plt.xlabel('Route type', labelpad=-1.5)
plt.ylabel('log(RouteScore) (a.u.)')
plt.tight_layout()
plt.savefig('Figure_6.pdf')
plt.show()
