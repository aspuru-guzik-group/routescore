import os
import pandas as pd
import datetime

HERE = os.path.abspath(os.path.dirname(__file__))
RSLT_DIR = os.path.join(HERE, 'Results')
TGT_DIR = os.path.join(HERE, 'Targets')
PROP_DIR = os.path.join(HERE, 'Properties')


props = pd.read_pickle(os.path.join(PROP_DIR, '20201221_full_props.pkl'))
newRS = pd.read_pickle(os.path.join(RSLT_DIR, 'All_RSonly.pkl'))

# props['route_score'] = [newRS[newRS['pentamer'] == smiles]['RouteScore'][0] for smiles in props['smiles']]

for i in range(len(props)):
    new_rs = newRS[newRS['pentamer'] == props.at[i, 'smiles']]['RouteScore']
    props.at[i, 'route_score'] = new_rs

today = datetime.datetime.today().strftime('%Y%m%d')
props.to_pickle(os.path.join(PROP_DIR, f'{today}_full_props.pkl'))
