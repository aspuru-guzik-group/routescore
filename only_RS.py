import os
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
RSLT_DIR = os.path.join(HERE, 'Results')
TGT_DIR = os.path.join(HERE, 'Targets')

base: pd.DataFrame = pd.read_pickle(os.path.join(TGT_DIR, 'targets_Base.pkl'))
s: pd.DataFrame = pd.read_pickle(os.path.join(TGT_DIR, 'targets_SNAr.pkl'))
b: pd.DataFrame = pd.read_pickle(os.path.join(TGT_DIR, 'targets_Buch.pkl'))
bs: pd.DataFrame = pd.read_pickle(os.path.join(TGT_DIR, 'targets_B-S.pkl'))
sb: pd.DataFrame = pd.read_pickle(os.path.join(TGT_DIR, 'targets_S-B.pkl'))

all_dfs = pd.DataFrame()
dfs = [base, s, b, bs]
df_names = ['Base', 'SNAr', 'Buch', 'B-S']
for df, name in zip(dfs, df_names):
    df = df[['pentamer', 'RouteScore', 'log(RouteScore)']]
    df.to_pickle(os.path.join(RSLT_DIR, f'{name}_RSonly.pkl'))
    all_dfs = all_dfs.append(df, ignore_index=True)

all_dfs.to_pickle(os.path.join(RSLT_DIR, 'All_RSonly.pkl'))