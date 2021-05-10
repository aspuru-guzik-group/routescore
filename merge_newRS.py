import os
import pandas as pd

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
RSLT_DIR = os.path.join(HERE, 'Results')
TGT_DIR = os.path.join(HERE, 'Targets')
PROP_DIR = os.path.join(HERE, 'Properties')

# Load quantum chemical properties
props = pd.read_pickle(os.path.join(PROP_DIR, 'qchem_props.pkl'))

# Load RouteScores
newRS = pd.read_pickle(os.path.join(RSLT_DIR, 'All_RSonly.pkl'))

# Update RouteScores
for i in range(len(props)):
    new_rs = newRS[newRS['pentamer'] == props.at[i, 'smiles']]['RouteScore']
    props.at[i, 'route_score'] = float(new_rs)

# Save dataframe of all properties
props.to_pickle(os.path.join(PROP_DIR, 'full_props.pkl'))
