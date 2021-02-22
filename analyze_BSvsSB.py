import os
from typing import Tuple, List
import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')


def extract_details(df):
    """Extract step details for last 3 steps (deBoc, BHA and SNAr)."""
    df_RSinfo = df[['pentamer', 'Step details', 'RouteScore details',
                    'Isolated', 'RouteScore', 'log(RouteScore)']]

    last3_rxns = ['Buchwald_deprotection', 'Buchwald', 'SNAr']
    for rxn in last3_rxns:
        df_RSinfo[rxn] = [next(step for step in row[-3:] if step['reaction'] == rxn) for row in df['Step details']]

    for key in df_RSinfo['RouteScore details'][0].keys():
        # df_RSinfo[key] = [[row[key] for key in row.keys()] for row in df['RouteScore details']]
        df_RSinfo[key] = [row[key] for row in df['RouteScore details']]

    return df_RSinfo


def plot_step_details(df1, df2, rxn, detail):
    """Plot comparison of details from B-S and S-B RouteScore."""
    a1, a2 = df1.align(df2, join='outer', axis=0)
    x_list = [row[detail] for row in a1[rxn]]
    y_list = [row[detail] for row in a2[rxn]]
    comparison_plot(x_list, y_list, f'{rxn} {detail}')


def plot_route_details(df1, df2, detail):
    """Plot comparison of RouteScore components from B-S and S-B."""
    a1, a2 = df1.align(df2, join='outer', axis=0)
    comparison_plot(df1[detail], df2[detail], detail)


def comparison_plot(X, Y, title):
    """General function for comparison plots."""
    linreg = scipy.stats.linregress(X, Y)

    midX = ((max(X) - min(X)) / 2) + min(X)
    midY = ((max(Y) - min(Y)) / 2) + min(Y)

    plt.scatter(X, Y)
    plt.xlabel('B-S')
    plt.ylabel('S-B')
    plt.title(title)
    plt.text(midX, midY, f'slope: {linreg.slope}\nR^2: {linreg.rvalue}')
    plt.ticklabel_format(axis='both', style='plain', useOffset=False)
    plt.show()
    print(f'Number of points:\nX: {len(X)}    Y: {len(Y)}')


bs = pd.read_pickle(os.path.join(TGT_DIR, 'targets_B-S.pkl'))
sb = pd.read_pickle(os.path.join(TGT_DIR, 'targets_S-B.pkl'))

# bs_RSinfo = pd.DataFrame()
bs_RSinfo = extract_details(bs)
# sb_RSinfo = pd.DataFrame()
sb_RSinfo = extract_details(sb)

# for step_detail in ['StepScore', 'cost', 'time', 'money', 'materials', 'distance']:
for rxn in ['Buchwald_deprotection', 'Buchwald', 'SNAr']:
    plot_step_details(bs_RSinfo, sb_RSinfo, rxn, 'distance')

print('\n––––––––––––––––––––––––––––––––––––\n')

# for route_detail in ['RouteScore', 'Drxn factor', 'max Drxn', 'avg Drxn',
#                      'Cost factor', 'sum Stepscores', 'n_Target']:
for route_detail in ['log(RouteScore)', 'Drxn factor', 'max Drxn', 'avg Drxn']:
    plot_route_details(bs_RSinfo, sb_RSinfo, route_detail)
