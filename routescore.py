import os
import pickle
from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import Descriptors, Draw
from IPython.display import display
IPythonConsole.ipython_useSVG = True

HERE = os.path.abspath(os.path.dirname(__file__))
SMRY_DIR = os.path.join(HERE, 'Rxn_Summs')
TYPE_DIR = os.path.join(HERE, 'Rxn_Type')
INV_DIR = os.path.join(HERE, 'Inventory')
PROP_DIR = os.path.join(HERE, 'Properties')

props_dict: dict = pickle.load(open(os.path.join(PROP_DIR, 'spectra_dict.pkl'), 'rb'))

# TODO: Add comprehensive docstrings for each function.

class Calculate:
    """Class containing operations for calculating StepScore."""

    rates_path = os.path.join(SMRY_DIR, 'hrly_costs.pkl')
    hrly_rates = pickle.load(open(rates_path, 'rb'))

    C_H: float = hrly_rates['C_H']
    C_M: float = hrly_rates['C_M']
    a: float = C_H / C_M
    a_: float = 1 / a

    inv_path = os.path.join(INV_DIR, 'Inventory.csv')
    inv = None
    inv = pd.read_csv(inv_path)

    def load_pkl(self, folder: str, file: str):
        """Load a .pkl file."""
        load_path = os.path.join(HERE, folder, f'{file}.pkl')
        file = pickle.load(open(load_path, 'rb'))
        return file

    def get_frag_info(self, frag_smiles: str) -> dict:
        """Get fragment informtion dict from inventory dataframe."""
        frag_entry = self.inv[self.inv['SMILES'] == frag_smiles]

        # print('get frag smiles:', frag_smiles)
        # self.draw_mols([frag_smiles])

        frag_dict = frag_entry.to_dict(orient='records')[0]
        return frag_dict

    def update_inventory(self, mol_smiles: str, mol_dict: dict) -> None:
        """Add new molecule to inventory if not already present."""
        if mol_smiles not in self.inv.SMILES.values:
            self.inv = self.inv.append(mol_dict, ignore_index=True)

    def draw_mols(self, mol_list: list) -> None:
        """Draw molecular structures inline."""
        for i in range(len(mol_list)):
            mol_list[i] = Chem.MolFromSmiles(mol_list[i])

        img = Draw.MolsToGridImage(mol_list, molsPerRow=3)
        display(img)

    def _stoichio(self, sms: List[dict], rxn: str) -> int:
        """Return the equivalent multiplier based on type of reaction."""
        patts: dict = {'Suzuki': Chem.MolFromSmiles('CBr'),
                       'Buchwald_deprotection': Chem.MolFromSmiles('O=C(OC(C)(C)C)n1c2c(cc1)cccc2'),
                       'Buchwald': Chem.MolFromSmiles('[H]n1c2c(cc1)cccc2'),
                       'SNAr': Chem.MolFromSmarts('cF')
                       }
        n_eqs: int = max([len(Chem.MolFromSmiles(sm['SMILES']).GetSubstructMatches(patts[rxn])) for sm in sms])
        # print('multiplier:', n_eqs)
        return n_eqs

    def _time(self, time_H: float, time_M: float) -> float:
        """Calculate temporal cost of reaction."""
        # print('t_H:', time_H)
        # print('t_M:', time_M)
        # print('a:', self.a)
        # print('_a:', self.a_)
        cost_t = np.sqrt((self.a * time_H)**2 + (self.a_ * time_M)**2)
        # print('cost_t:', cost_t)
        # cost_t = 2 * self.a * (1 / time_H) + self.a_ * (1/ time_M)

        return cost_t

    def _money(self,
               reagents: List[dict],
               sm_costs: List[float],
               sm_eqs: List[float],
               time_H: float,
               time_M: float,
               rxn_scale: float,
               multiplier: int
               ) -> float:
        """Calculate the monetary cost of the reaction."""
        rgt_cost = 0
        rgt_cost: float = sum([rgt['$/mol'] * rgt['eq'] * multiplier for rgt in reagents])
        rct_cost: float = 0
        # rct_cost: float = sum([cost * eq for cost, eq in zip(sm_costs, sm_eqs)])
        for cost, eq in zip(sm_costs, sm_eqs):
            if eq == 1:
                rct_cost += cost * eq
            else:
                rct_cost += cost * eq * multiplier
        mater_cost: float = rxn_scale * (rct_cost + rgt_cost)
        # mater_cost = mol1_cost * mol1_eq * rxn_scale + mol2_cost * mol2_eq * rxn_scale + rgt_cost

        cost_cad: float = mater_cost + (time_H * self.C_H + time_M * self.C_M)

        return cost_cad

    def _materials(self,
                   reagents: List[dict],
                   sm_mws: List[float],
                   sm_eqs: List[float],
                   rxn_scale: float,
                   multiplier: int
                   ) -> float:
        """Calculate the material cost of the reaction."""
        rgt_quant: float = sum([rgt['g/mol'] * rgt['eq'] * multiplier for rgt in reagents])
        rct_quant: float = 0
        # rct_quant: float = sum([mw * eq for mw, eq in zip(sm_mws, sm_eqs)])
        for mw, eq in zip(sm_mws, sm_eqs):
            if eq == 1:
                rct_quant += mw * eq
            else:
                rct_quant += mw * eq * multiplier
        cost_mat: float = rxn_scale * (rct_quant + rgt_quant)
        # cost_mat = mol1_mw * mol1_eq * rxn_scale + mol2_mw * mol2_eq * rxn_scale + rgt_quant

        return cost_mat

    def _similarity(self,
                    sm_smiles: str,
                    pdt_smiles: str,
                    tgt_smiles: str
                    ) -> float:
        """Calculate distance travelled by reaction using fingerprint sims."""
        # mol_list = [sm_smiles, pdt_smiles, tgt_smiles]
        # for i in range(len(mol_list)):
        #     mol_list[i] = Chem.MolFromSmiles(mol_list[i])

        # img = Draw.MolsToGridImage(mol_list, molsPerRow=3)
        # display(img)

        sim_BC = self.fp_similarity(pdt_smiles, tgt_smiles)
        sim_AC = self.fp_similarity(sm_smiles, tgt_smiles)

        if sim_BC < sim_AC:
            print('sim_BC:', sim_BC)
            print('sim_AC:', sim_AC)
            print('!!!\nATTN: Product less similar than SM!\n!!!')
            mol_list = [sm_smiles, pdt_smiles, tgt_smiles]
            mol_list = [Chem.MolFromSmiles(mol) for mol in mol_list]
            img = Draw.MolsToGridImage(mol_list, molsPerRow=3)
            display(img)

        if sim_AC == 0:
            sim_AC = 0.0000000001
        travel = sim_BC / (sim_AC)

        return travel

    def fp_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate fingerprint simuilarity of 2 molecules."""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        assert mol1 is not None, f"Invalid smiles {smiles1}"
        assert mol2 is not None, f"Invalid smiles {smiles2}"

        minP = 1
        maxP = 7
        fpSize = 8192

        fp1 = RDKFingerprint(mol1, minPath=minP, maxPath=maxP, fpSize=fpSize)
        fp2 = RDKFingerprint(mol2, minPath=minP, maxPath=maxP, fpSize=fpSize)

        return DataStructs.FingerprintSimilarity(fp1, fp2)

    def get_man_synth(self, sm_smiles: str, tgt_smiles: str, rxn_scale: float) -> Tuple[float, float, List[float]]:
        """Compare input to known list of 'manual' molecules and return score and $/mol."""
        lookup_df = pd.DataFrame()
        lookup_df = pd.read_excel(os.path.join(SMRY_DIR, 'ManSynth_lookup.xlsx'))
        smry_fname: str = lookup_df['summary file'][lookup_df['SMILES'] == sm_smiles].iloc[0]
        man_df: dict = pd.read_excel(os.path.join(SMRY_DIR, smry_fname), sheet_name=None, dtype={'SM?': bool})

        l_man_dist = []
        score: float = 0
        for step in man_df:
            t_H: float = man_df[step]['t_H'][0]
            t_M: float = man_df[step]['t_M'][0]
            # man_yld: float = man_df[step]['yield'][0]
            man_yld: float = 1
            man_pdt_smiles: str = man_df[step]['pdt_smiles'][0]

            man_scale: float = rxn_scale / man_yld
            # man_t: float = 1 + np.sqrt(self.a * t_H**2 + self.a_ * t_M**2)
            man_time: float = self._time(t_H, t_M)
            # print('time cost:', man_t)
            man_money: float = ((self.C_H * t_H + self.C_M * t_M) + sum(man_df[step]['$/mol'] * man_df[step]['eq'] * man_scale))
            # print('$/mol:', man_d)
            man_materials: float = sum(man_df[step]['g/mol'] * man_df[step]['eq'] * man_scale)
            # print('mtrl cost:', man_m)
            cost: float = man_time * man_money * man_materials

            manSM_df = man_df[step][man_df[step]['SM?'] == True]
            # SMs_list: List[str] = [row for row in manSM_df['SMILES']]
            # self.draw_mols(SMs_list)
            # self.draw_mols([man_pdt_smiles, tgt_smiles])
            sims_list: List[float] = [self._similarity(row, man_pdt_smiles, tgt_smiles) for row in manSM_df['SMILES']]
            man_dist: float = min(sims_list)
            l_man_dist.append(man_dist)

            score += cost
            man_molarCost = man_yld * man_money / man_scale
            # print('step cost:', cost)
            # print('step distance:', man_dist)
            # print('Man score:', score)

        return score, man_molarCost, l_man_dist

    def StepScore(self,
                  sm_list: List[dict],
                  product_smiles: str,
                  target_smiles: str,
                  rxn_type: str,
                  scale: float,
                  yld: float,
                  ) -> float:
        """Perform calculations for the StepScore."""
        man_stepscore = 0
        mandists_list = []

        frag_dicts: List[dict] = [self.get_frag_info(sm['smiles']) for sm in sm_list]
        # print(frag_dicts)

        reaction: list = self.load_pkl(TYPE_DIR, rxn_type)
        rxn_smry: dict = self.load_pkl(SMRY_DIR, f'{rxn_type}_summary')

        # Number of parallel reactions by robot
        n_parr: int = rxn_smry['n_parr']
        t_H: float = rxn_smry['t_H'] / n_parr
        t_M: float = rxn_smry['t_M'] / n_parr
        # yld: float = rxn_smry['yield']
        # scale: float = rxn_smry['scale']

        # man_pre = ManualPreSynthesis()
        # lookup_df: pd.DataFrame = pd.read_excel(os.path.join(SMRY_DIR, 'ManSynth_lookup.xlsx'))

        man_frags: List[dict] = [frag for frag in frag_dicts if frag['$/mol'] == 0]
        if len(man_frags) > 0:
            print('Manual fragments:', man_frags)

        # if any(molecule in man_frags for molecule in lookup_df['SMILES']):
        for frag in man_frags:
            man_mol = {'score': 0, '$/mol': 0}
            print('ATTN:    Manually synthesized starting material!')
            self.draw_mols([frag['SMILES']])
            man_mol['score'], frag['$/mol'], mandists_list = self.get_man_synth(frag['SMILES'],
                                                                                target_smiles,
                                                                                scale)
            # print(f"Added {man_mol['score']} to StepScore.")
            # print(f"Cost of manual fragment is {frag['$/mol']} $/mol.")
            man_stepscore += man_mol['score']

        sm_eqs: List[float] = [sm['eq'] for sm in sm_list]
        eq_mult: int = self._stoichio(frag_dicts, rxn_type)

        # Cost to travel through chemical space
        cost_time: float = self._time(t_H, t_M)
        # print('cost_time:', cost_time)
        frag_costs: List[float] = [sm['$/mol'] for sm in frag_dicts]
        cost_money: float = self._money(reaction,
                                        frag_costs,
                                        sm_eqs,
                                        t_H,
                                        t_M,
                                        scale,
                                        eq_mult)
        # t_money: float = t_H * self.C_H + t_M * self.C_M
        # m_money: float = cost_money - t_money
        # print('cost_money:', cost_money)
        frag_mws: List[float] = [sm['g/mol'] for sm in frag_dicts]
        cost_materials: float = self._materials(reaction,
                                                frag_mws,
                                                sm_eqs,
                                                scale,
                                                eq_mult)
        # print('cost_materials:', cost_materials)
        cost: float = cost_time * cost_money * cost_materials

        # Distance "traveled" in chemical space by reaction
        sims_list: List[float] = [self._similarity(sm['SMILES'], product_smiles, target_smiles) for sm in frag_dicts]

        distance: float = min(sims_list)

        step_score: float = cost
        # print('StepScore w/o ManSynth:', step_score)
        step_score += man_stepscore
        # print('Cost:', cost)
        # print('Distance:', distance)
        # print('Manual StepScore:', man_stepscore)
        # print('StepScore:', step_score)

        MW: float = Descriptors.MolWt(Chem.MolFromSmiles(product_smiles))

        molarCost = yld * cost_money / scale

        product_dict: dict = {
            'Frag_type': '-',
            'Frag_num': 0,
            'SMILES': product_smiles,
            'Name': '-',
            'g/mol': MW,
            'Quantity': 0,
            'CAD': 0,
            '$/mol': molarCost
            }

        self.update_inventory(product_smiles, product_dict)

        self.inv.to_csv(self.inv_path, index=False)

        stepscore_results: dict = {
            'StepScore': step_score,
            'cost': cost,
            'time': cost_time,
            'money': cost_money,
            'materials': cost_materials,
            'yield': yld,
            'distance': distance,
            'man rxn distances': mandists_list
            }

        return stepscore_results

    def RouteScore(self, steps_list: List[float], total_yield: float) -> float:
        """Calculate the total RouteScore."""
        stepscores_list: List[float] = [step['StepScore'] for step in steps_list]
        distances_list: List[float] = [step['distance'] for step in steps_list]
        mandists_nested: List[List[float]] = [step['man rxn distances'] for step in steps_list]
        mandists_l: List[float] = [item for sublist in mandists_nested for item in sublist]
        all_distances: List[float] = distances_list + mandists_l

        cost_factor: float = sum(stepscores_list) / total_yield
        distance_factor: float = max(all_distances) / np.mean(all_distances)
        route_score: float = cost_factor / distance_factor
        return route_score


class Properties:
    """Class for getting the predicted properties of molecules."""

    def get_props(self, routes_df: pd.DataFrame):
        """Extract relevant properties from the properties dict."""
        no_prop_mols = 0
        for i in range(len(routes_df.index)):
            smiles: str = routes_df.at[i, 'pentamer']
            # print('smiles', smiles)
            score: float = routes_df.at[i, 'RouteScore']
            if np.isnan(score):
                print('ATTN! Score is NaN!', score)

            try:
                spectra: pd.DataFrame = props_dict[smiles]

                ab: pd.Series = spectra['extinct']
                em: pd.Series = spectra['fluo']

                amax_nm: float = spectra.at[spectra['extinct'].idxmax(), 'x_nm']
                emax_nm: float = spectra.at[spectra['fluo'].idxmax(), 'x_nm']

                ovlp: float = self.overlap(ab, em)
                if np.isnan(ovlp):
                    print('ATTN! Overlap is NaN!', ovlp)

                routes_df.at[i, 'Overlap'] = ovlp
                routes_df.at[i, '1 / Overlap'] = 1 / ovlp
                routes_df.at[i, 'Abs max'] = amax_nm
                routes_df.at[i, 'Em max'] = emax_nm

            except KeyError:
                # print('ATTN! Molecule(s) not in dictionary.')
                # print('Molecule:', smiles)
                no_prop_mols += 1
                # display(Chem.MolFromSmiles(smiles))

        pct_no_props = round(100 * no_prop_mols / len(routes_df.index), 1)
        print(f'ATTN: {pct_no_props}% of molecules ({no_prop_mols} / {len(routes_df.index)}) not in properties dictionary!')

        return routes_df

    def overlap(self, y1: pd.Series, y2: pd.Series) -> float:
        """Calculate emission and absorbance overlap."""
        return np.sum(y1 * y2) / np.sqrt(np.sum(y1 * y1) * np.sum(y2 * y2))


class Analysis:
    """Class for analyzing RouteScore results."""

    def plotting(self, results: pd.DataFrame):
        """Produce relevant plots for quick data analysis."""
        plt.scatter(results['log(RouteScore)'], results['1 / Overlap'])
        plt.ylabel('1 / Overlap')
        plt.xlabel('log(RouteScore)')
        plt.show()

        print('\nSmallest spectral overlap')
        ovlp_mol = results.loc[results['Overlap'].idxmin(), 'pentamer']
        display(Chem.MolFromSmiles(ovlp_mol))
        norm_ab: pd.Series = props_dict[ovlp_mol]['extinct'] / max(props_dict[ovlp_mol]['extinct'])
        norm_em: pd.Series = props_dict[ovlp_mol]['fluo'] / max(props_dict[ovlp_mol]['fluo'])
        nm: pd.Series = props_dict[ovlp_mol]['x_nm']
        #ev: pd.Series = props_dict[ovlp_mol]['x_ev']
        plt.plot(nm, norm_ab, label='Absorbance')
        plt.plot(nm, norm_em, label='Emission')
        plt.xlim(200, 1200)
        plt.ylabel('Normalized intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.legend()
        plt.show()
