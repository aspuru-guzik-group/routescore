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

c_list = [
          'Brc1ccc2oc(Br)cc2c1',
          'C[Si](C)(c1ccc(Br)cc1)c1ccc(Br)cc1',
          'C[Si](C)(c1ccccc1Br)c1ccccc1Br'
          ]

print('base:', len(base))
base_man = pd.DataFrame()
base_man = base[base.c.isin(c_list)]
base = base[~base.c.isin(c_list)]
print('base auto:', len(base))
print('base man', len(base_man))

print('b:', len(b))
b_man = pd.DataFrame()
b_man = b[b.c.isin(c_list)]
b = b[~b.c.isin(c_list)]
print('b auto:', len(b))
print('b man', len(b_man))

print('s:', len(s))
s_man = pd.DataFrame()
s_man = s[s.c.isin(c_list)]
s = s[~s.c.isin(c_list)]
print('s auto:', len(s))
print('s man', len(s_man))

print('bs:', len(bs))
bs_man = pd.DataFrame()
bs_man = bs[bs.c.isin(c_list)]
bs = bs[~bs.c.isin(c_list)]
print('bs auto:', len(bs))
print('bs man', len(bs_man))

print('sb:', len(sb))
sb_man = pd.DataFrame()
sb_man = sb[sb.c.isin(c_list)]
sb = sb[~sb.c.isin(c_list)]
print('sb auto:', len(sb))
print('sb man', len(sb_man))

base['Set'] = 'Base'
base['Type'] = 'Automatic'
base['Manual'] = 'None'
base['Manual steps'] = 0
base_man['Set'] = 'Base'
base_man['Type'] = 'Manual'
base_man['Manual'] = 'Before'
base_man['Manual steps'] = 1
b['Set'] = 'Buchwald'
b['Type'] = 'Automatic'
b['Manual'] = 'After'
b['Manual steps'] = 1
b_man['Set'] = 'Buchwald'
b_man['Type'] = 'Manual'
b_man['Manual'] = 'Both'
b_man['Manual steps'] = 2
s['Set'] = 'SNAr'
s['Type'] = 'Automatic'
s['Manual'] = 'After'
s['Manual steps'] = 1
s_man['Set'] = 'SNAr'
s_man['Type'] = 'Manual'
s_man['Manual'] = 'Both'
s_man['Manual steps'] = 2
bs['Set'] = 'B-S'
bs['Type'] = 'Automatic'
bs['Manual'] = 'After'
bs['Manual steps'] = 2
bs_man['Set'] = 'B-S'
bs_man['Type'] = 'Manual'
bs_man['Manual'] = 'Both'
bs_man['Manual steps'] = 3
sb['Set'] = 'S-B'
sb['Type'] = 'Automatic'
sb['Manual'] = 'After'
sb['Manual steps'] = 2
sb_man['Set'] = 'S-B'
sb_man['Type'] = 'Manual'
sb_man['Manual'] = 'Both'
sb_man['Manual steps'] = 3

dfs = [base, base_man, s, s_man, b, b_man, bs, bs_man, sb, sb_man]
df_names = ['Base', 'Base_man', 'SNAr', 'SNAr_man', 'Buch', 'Buch_man', 'B-S', 'B-S_man', 'S-B', 'S-B_man']
for df, name in zip(dfs, df_names):
    df.to_pickle(os.path.join(RSLT_DIR, f'{name}.pkl'))
