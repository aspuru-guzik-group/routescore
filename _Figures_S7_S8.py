import os
from typing import Tuple, List
import pandas as pd
import scipy.stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Custom style
mpl.style.use('scientific')

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
TGT_DIR = os.path.join(HERE, 'Targets')
FIG_DIR = os.path.join(HERE, 'Figures_S7_S8')


def extract_details(df):
    """Extract step details for last 3 steps (deBoc, BHA and SNAr)."""
    df_RSinfo = df[['pentamer', 'Step details', 'RouteScore details',
                    'Isolated', 'RouteScore', 'log(RouteScore)']]

    last3_rxns = ['Buchwald_deprotection', 'Buchwald', 'SNAr']
    for rxn in last3_rxns:
        df_RSinfo[rxn] = [next(step for step in row[-3:] if step['reaction'] == rxn) for row in df['Step details']]

    for key in df_RSinfo['RouteScore details'][0].keys():
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

    x_txt = rel_pos(X, 0.95)
    y_txt = rel_pos(Y, 0)

    plt.scatter(X, Y)
    plt.xlabel('B-S')
    plt.ylabel('S-B')
    if title == 'RouteScore':
        plt.xscale('log')
        plt.yscale('log')
    if len(set(X)) > 1 or len(set(Y)) > 1:
        plt.text(x_txt, y_txt,
                 f'slope = {round(linreg.slope, 2)}\n$R^{2}$ = {round(linreg.rvalue,2)}',
                 horizontalalignment='right',
                 verticalalignment='bottom')
    plt.xlim(0.95 * min(X), 1.05 * max(X))
    plt.ylim(0.95 * min(Y), 1.05 * max(Y))
    plt.savefig(os.path.join(FIG_DIR, f'BSvsSB {title}.png'))
    plt.show()


def rel_pos(data, pct) -> float:
    return min(data) + pct * (max(data) - min(data))


# Load B-S and S-B route dataframes
bs = pd.read_pickle(os.path.join(TGT_DIR, 'targets_B-S.pkl'))
sb = pd.read_pickle(os.path.join(TGT_DIR, 'targets_S-B.pkl'))

# Extract RouteScore details
bs_RSinfo = pd.DataFrame()
bs_RSinfo = extract_details(bs)
sb_RSinfo = pd.DataFrame()
sb_RSinfo = extract_details(sb)

# Plots for Figure S7
for step_detail in ['time', 'money', 'materials']:
    for rxn in ['Buchwald_deprotection', 'Buchwald', 'SNAr']:
        plot_step_details(bs_RSinfo, sb_RSinfo, rxn, step_detail)

# Figure S8
plot_route_details(bs_RSinfo, sb_RSinfo, 'RouteScore')
