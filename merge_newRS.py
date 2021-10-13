import os
import pandas as pd

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
RSLT_DIR = os.path.join(HERE, 'Results')
TGT_DIR = os.path.join(HERE, 'Targets')
PROP_DIR = os.path.join(HERE, 'Properties')

# Load quantum chemical properties
props = pd.read_pickle(os.path.join(PROP_DIR, 'props_noRS.pkl'))

# Load RouteScores
newRS = pd.read_pickle(os.path.join(RSLT_DIR, 'All_RSonly.pkl'))

# Update RouteScores
for i in range(len(props)):
    new_rs: pd.Series = newRS[newRS['pentamer'] == props.at[i, 'smiles']]['RouteScore']
    new_naive: pd.Series = newRS[newRS['pentamer'] == props.at[i, 'smiles']]['NaiveScore']
    props.at[i, 'route_score'] = float(new_rs)
    props.at[i, 'naive_score'] = float(new_naive)

# Save dataframe of all properties
props.to_pickle(os.path.join(PROP_DIR, 'full_props.pkl'))
