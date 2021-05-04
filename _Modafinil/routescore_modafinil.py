import numpy as np
import pandas as pd

# Modafinil SMILES string
modaf_SMILES = 'O=S(CC(N)=O)C(C1=CC=CC=C1)C2=CC=CC=C2'

n_routes = 10
for i in range(1, n_routes+1):
    print(f'Route {i}')
    fname = f'Route_{i}.xlsx'
    route_xls = pd.read_excel(fname, sheet_name=None)

    C_time = []
    C_money = []
    C_mass = []
    step_scores = []

    for step in route_xls.keys():
        print(step)
        C_time.append(route_xls[step]['C_time'][0])
        C_money.append(route_xls[step]['C_money'][0])
        C_mass.append(route_xls[step]['C_mass'][0])
        step_scores.append(route_xls[step]['raw SS'][0])
        target_mols = route_xls[step]['n_prod'][0]

    n_steps = len(step_scores)
    avg_time = sum(C_time) / n_steps
    avg_money = sum(C_money) / n_steps
    avg_mass = sum(C_mass) / n_steps
    tot_time = sum(C_time)
    tot_yield = (target_mols / (route_xls['Step 1']['n_prod'][0] / route_xls['Step 1']['Yield'][0])) * 100
    step_sums = sum(step_scores)
    cost = step_sums / target_mols
    route_score = cost

    route_xls['Summary'] = pd.DataFrame(columns=[
                                                 '# steps',
                                                 'total C_time',
                                                 'avg C_time',
                                                 'avg C_money',
                                                 'avg C_mass',
                                                 'SUM raw SS',
                                                 'n_Target',
                                                 'cost factor',
                                                 'full RS',
                                                 'log RS'
                                                 ]
                                        )
    route_xls['Summary'].loc[0, '# steps'] = n_steps
    route_xls['Summary'].loc[0, 'total C_time'] = tot_time
    route_xls['Summary'].loc[0, 'avg C_time'] = avg_time
    route_xls['Summary'].loc[0, 'avg C_money'] = avg_money
    route_xls['Summary'].loc[0, 'avg C_mass'] = avg_mass
    route_xls['Summary'].loc[0, 'SUM raw SS'] = step_sums
    route_xls['Summary'].loc[0, 'n_Target'] = target_mols
    route_xls['Summary'].loc[0, 'Total yield'] = tot_yield
    route_xls['Summary'].loc[0, 'cost factor'] = cost
    route_xls['Summary'].loc[0, 'full RS'] = route_score
    route_xls['Summary'].loc[0, 'log RS'] = np.log10(route_score)

    with pd.ExcelWriter(f'output_Route_{i}.xlsx') as writer:
        for step in route_xls.keys():
            route_xls[step].to_excel(writer, sheet_name=step, index=False)
