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

# Directories
HERE = os.path.abspath(os.path.dirname(__file__))
SMRY_DIR = os.path.join(HERE, 'Rxn_Summs')
TYPE_DIR = os.path.join(HERE, 'Rxn_Type')
INV_DIR = os.path.join(HERE, 'Inventory')
PROP_DIR = os.path.join(HERE, 'Properties')

# Files
props_dict: dict = pickle.load(open(os.path.join(PROP_DIR, 'spectra_dict.pkl'), 'rb'))
inv_path = os.path.join(INV_DIR, 'Inventory.csv')


class General:
    """Class for general functions and data management operations."""

    def load_pkl(self, folder: str, file: str):
        """Load a .pkl file.

        Parameters
        ----------
        folder:     Directory where the file is located. Must be within HERE.
        file:       Filname of the .pkl file

        Returns
        -------
        pkl: Contents of the .pkl file
        """
        load_path = os.path.join(HERE, folder, f'{file}.pkl')
        pkl = pickle.load(open(load_path, 'rb'))
        return pkl

    def draw_mols(self, smiles_list: List[str]) -> None:
        """Draw molecular structures inline.

        Parameters
        ----------
        mol_list: List of SMILES to draw

        Returns
        -------
        Nothing
        """
        # for i in range(len(mol_list)):
        #     mol_list[i] = Chem.MolFromSmiles(mol_list[i])
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

        img = Draw.MolsToGridImage(mol_list, molsPerRow=3)
        display(img)

    def preProcess(self, df_file: os.PathLike):
        """Prepare dataframe with routes for details dictionaries.

        Parameters
        ----------
        df_file: Path for the file containing the dataframe

        Returns
        -------
        df: Appropriately formatted dataframe.
        """
        df: pd.DataFrame = pd.read_pickle(df_file)
        df['Step details'] = ''
        df['Step details'] = df['Step details'].astype('object')
        df['RouteScore details'] = ''
        df['RouteScore details'] = df['RouteScore details'].astype('object')
        return df

    def Process(self,
                targets_df: pd.DataFrame,
                i: int,
                steps_list: List[float],
                n_target: float
                ) -> pd.DataFrame:
        """Do some final processing for the route in the full dataframe.

        Adds the following data:
            Isolated: mass of isolated target molecule
            RouteScore: the RouteScore
            log(RouteScore): log10 of the RouteScore
            Step details: list of all stepscore_results dictionaries
            Total manual steps: total number of manual steps in the route

        Parameters
        ----------
        targets_df: Full dataframe of all target molecules
        i:          Index of the target molecule in the df
        steps_list: List containing details for all the steps along the route
        n_target:   Quantity (mols) of the target molecule obtained

        Returns
        -------
        targets_df: Updated full dataframe of all target molecules
        """
        isolated: float = n_target * Descriptors.MolWt(Chem.MolFromSmiles(targets_df.at[i, 'pentamer']))
        # print(f'Isolated yield:    {isolated} g')
        routescore_details, total_man = Calculate().RouteScore(steps_list, n_target)
        # print(total_man)

        targets_df.at[i, 'Isolated'] = isolated
        targets_df.at[i, 'RouteScore'] = routescore_details['RouteScore']
        targets_df.at[i, 'log(RouteScore)'] = np.log10(routescore_details['RouteScore'])
        targets_df.at[i, 'Step details'] = steps_list
        targets_df.at[i, 'RouteScore details'] = routescore_details
        targets_df.at[i, 'Total manual steps'] = int(total_man)
        targets_df.at[i, 'Successfully processed?'] = True
        return targets_df


class Reaction_Templates:
    """Class containing centralized templates for all the reactions."""

    # calc = Calculate()

    def stoichiometry(self, smiles: str, rxn: str) -> int:
        """Determine the number of reaction sites.

        Parameters
        ----------
        smiles:    SMILES of the molecule with reaction site substructures
        rxn:    Name of reaction to match to dictionary with substructures to count

        Returns
        -------
        mult: Number of reaction sites on the molecule
        """
        # TODO: Load 'patts' from an external .pkl file.
        patts: dict = {'Suzuki': [Chem.MolFromSmarts('cBr'),
                                  Chem.MolFromSmarts('cI')],
                       'deBoc': [Chem.MolFromSmiles('O=C(OC(C)(C)C)n1c2c(cc1)cccc2')],
                       'BHA-H': [Chem.MolFromSmiles('[H]n1c2c(cc1)cccc2')],
                       'BHA-Py': [Chem.MolFromSmiles('n1(c2nccnc2)ccc3ccccc13')],
                       'SNAr-F': [Chem.MolFromSmarts('cF')],
                       'SNAr-Cz': [Chem.MolFromSmarts('cn(c1c2cccc1)c3c2cccc3')]
                       }
        mol = Chem.MolFromSmiles(smiles)
        matches_list: List[int] = [len(mol.GetSubstructMatches(substruct))
                                   for substruct in patts[rxn]]
        # mult: int = len(Chem.MolFromSmiles(mol).GetSubstructMatches(patts[rxn]))
        mult: int = sum(matches_list)
        # print(f'{rxn} multiplier:', mult)
        if mult == 0:
            General().draw_mols([smiles])
        return mult

    def wingSuzuki(self, smilesA: str, smilesB: str, smilesAB: str, smilesTgt: str, scale: float) -> Tuple[dict, float]:
        """Template for Suzuki coupling of the A-B wing.

        Parameters
        ----------
        smilesA:   SMILES of the A reactant molecule
        smilesB:   SMILES of the B reactant molecule
        smilesAB:  SMILES of the product (A-B) molecule
        smilesTgt: SMILES of the target molecule
        scale:  scale of the reaction (mol)

        Returns
        -------
        Calculate().StepScore()
        wS_scale * wS_yield:    scale for next step
        """
        sm_list: List[dict] = [{'smiles': smilesA, 'eq': 3},
                               {'smiles': smilesB, 'eq': 1}
                               ]
        rxn_sites = self.stoichiometry(smilesB, 'Suzuki')
        sm_list[0]['eq'] = rxn_sites * sm_list[0]['eq']
        # print('New eq:', sm_list[0]['eq'])
        wS_scale = scale
        wS_yield: float = 1
        return Calculate().StepScore(
                                     sm_list,
                                     smilesAB,
                                     smilesTgt,
                                     'Suzuki',
                                     rxn_sites,
                                     wS_scale,
                                     wS_yield,
                                     False
                                     ), wS_scale * wS_yield

    def pentamerSuzuki(self, smilesAB: str, smilesC: str, smilesABCBA: str, smilesTgt: str, scale: float) -> Tuple[dict, float]:
        """Template for Suzuki coupling of the A-B-C-B-A pentamer.

        Parameters
        ----------
        smilesAB:      SMILES of the A-B wing reactant molecule
        smilesC:       SMILES of the C reactant molecule
        smilesABCBA:   SMILES of the product (A-B-C-B-A) molecule
        smilesTgt:     SMILES of the target molecule
        scale:  scale of the reaction (mol)

        Returns
        -------
        Calculate().StepScore()
        pS_scale * pS_yield:    scale for next step
        """
        sm_list: List[dict] = [
                               {'smiles': smilesAB, 'eq': 3},    # This should be checked...
                               {'smiles': smilesC, 'eq': 1}
                               ]
        rxn_sites = self.stoichiometry(smilesC, 'Suzuki')
        sm_list[0]['eq'] = rxn_sites * sm_list[0]['eq']
        # print('New eq:', sm_list[0]['eq'])
        # Here we are dividing by eq of 'ab' because it's assumed that it is the limiting reagent
        # since it's the only synthesized compound.
        pS_scale: float = scale / sm_list[0]['eq']
        pS_yield: float = 1
        return Calculate().StepScore(
                                     sm_list,
                                     smilesABCBA,
                                     smilesTgt,
                                     'Suzuki',
                                     rxn_sites,
                                     pS_scale,
                                     pS_yield,
                                     False
                                     ), pS_scale * pS_yield

    def deBoc(self, smilesNBoc: str, smilesNH: str, smilesTgt: str, scale: float) -> Tuple[dict, float]:
        """Template for Boc-deprotection reaction.

        Parameters
        ----------
        smilesNBoc:    SMILES of the Boc-protected reactant molecule.
        smilesNH:      SMILES of the product (deprotected) molecule.
        smilesTgt:     SMILES of the target molecule
        scale:  scale of the reaction

        Returns
        -------
        Calculate().StepScore()
        dB_scale * dB_yield:    scale for next step
        """
        sm_list: List[dict] = [
                               {'smiles': smilesNBoc, 'eq': 1}
                               ]
        rxn_sites = self.stoichiometry(smilesNBoc, 'deBoc')
        # sm_list[0]['eq'] = rxn_sites * sm_list[0]['eq']
        # print('New eq:', sm_list[0]['eq'])
        dB_scale: float = scale / sm_list[0]['eq']
        dB_yield: float = 1
        return Calculate().StepScore(
                                     sm_list,
                                     smilesNH,
                                     smilesTgt,
                                     'Buchwald_deprotection',
                                     rxn_sites,
                                     dB_scale,
                                     dB_yield,
                                     True
                                     ), dB_scale * dB_yield

    def BHA(self, smilesNH: str, smilesX: str, smilesBHA: str, smilesTgt: str, scale: float) -> Tuple[dict, float]:
        """Template for Buchwald-Hartwig amination reaction.

        Parameters
        ----------
        smilesNH:      SMILES of the amine reactant molecule
        smilesX:       SMILES of the halide reactant molecule
        smilesBHA:     SMILES of the product molecule
        smilesTgt:     SMILES of the target molecule
        scale:      scale of the reaction

        Returns
        -------
        Calculate().StepScore()
        BHA_scale * BHA_yield:    scale for next step
        """
        sm_list: List[dict] = [
                               {'smiles': smilesNH, 'eq': 1},
                               {'smiles': smilesX, 'eq': 3}
                               ]
        rxn_sites = self.stoichiometry(smilesNH, 'BHA-H')
        # rxn_sites = self.stoichiometry(smilesBHA, 'BHA-Py')
        # print('rxn_sites:', rxn_sites)
        # if rxn_sites != 2:
        #     print('BHA reaction sites not 2!', rxn_sites)
        sm_list[1]['eq'] = rxn_sites * sm_list[1]['eq']
        # print('New eq:', sm_list[1]['eq'])
        BHA_scale: float = scale / sm_list[0]['eq']
        BHA_yield: float = 1
        return Calculate().StepScore(
                                     sm_list,
                                     smilesBHA,
                                     smilesTgt,
                                     'Buchwald',
                                     rxn_sites,
                                     BHA_scale,
                                     BHA_yield,
                                     True
                                     ), BHA_scale * BHA_yield

    def SNAr(self, smilesAr: str, smilesNu: str, smilesSNAr: str, smilesTgt: str, scale: float) -> Tuple[dict, float]:
        """Template for nucleophilic aromatic substitution reaction.

        Parameters
        ----------
        smilesAr:      SMILES of aromatic reactant molecule
        smilesNu:      SMILES of nucleophile reactant molecule
        smilesSNAr:    SMILES of product molecule
        smilesTgt:     SMILES of target molecule
        scale:      scale of the reaction

        Returns
        -------
        Calculate().StepScore()
        SNAr_scale * SNAr_yield:    scale for next step
        """
        sm_list: List[dict] = [
                               {'smiles': smilesAr, 'eq': 1},
                               {'smiles': smilesNu, 'eq': 2}
                               ]
        # HACK: In the case of SNAr reaction, we count the number of Cz groups in the product...
        # rxn_sites = self.stoichiometry(smilesAr, 'SNAr-F')
        rxn_sites = self.stoichiometry(smilesSNAr, 'SNAr-Cz')
        # print('rxn_sites:', rxn_sites)
        # if rxn_sites > 4:
            # print(f'ATTN! rxn_sites > 4!')
        sm_list[1]['eq'] = rxn_sites * sm_list[1]['eq']
        # print('New eq:', sm_list[1]['eq'])
        SNAr_scale: float = scale / sm_list[0]['eq']
        SNAr_yield: float = 1
        return Calculate().StepScore(
                                     sm_list,
                                     smilesSNAr,
                                     smilesTgt,
                                     'SNAr',
                                     rxn_sites,
                                     SNAr_scale,
                                     SNAr_yield,
                                     True
                                     ), SNAr_scale * SNAr_yield


class Calculate:
    """Class containing operations for calculating StepScore."""

    rates_path = os.path.join(SMRY_DIR, 'hrly_costs.pkl')
    hrly_rates = pickle.load(open(rates_path, 'rb'))

    C_H: float = hrly_rates['C_H']
    C_M: float = hrly_rates['C_M']
    a: float = C_H / C_M
    a_: float = 1 / a

    # inv_path = os.path.join(INV_DIR, 'Inventory.csv')
    # inv = None
    # inv = pd.read_csv(inv_path)

    def __init__(self):
        self.inv = None
        self.inv = pd.read_csv(inv_path)

    def get_block_info(self, smilesBlock: str) -> dict:
        """Get block informtion dict from inventory dataframe.

        Parameters
        ----------
        smilesBlock: SMILES of the building block to search for in inventory

        Returns
        -------
        block_info: Dictionary containing inventory information for smilesBlock molecule
        """
        # print(f'Inventory size (reading): {len(self.inv)}')
        block_entry = self.inv[self.inv['SMILES'] == smilesBlock]

        # General().draw_mols([smilesBlock])
        # print('get block smiles:', smilesBlock)

        # print(f'length of block_entry:    {len(block_entry)}')
        block_info: dict = block_entry.to_dict(orient='records')[0]

        return block_info

    def update_inventory(self, mol_smiles: str, mol_dict: dict) -> None:
        """Add new molecule to inventory if not already present.

        Parameters
        ----------
        mol_smiles: SMILES of molecule to add to inventory
        mol_dict:   Dictionary of new molecule's inventory data

        Returns
        -------
        Nothing
        """
        # orig_len = len(self.inv)
        if mol_smiles not in self.inv.SMILES.values:
            # General().draw_mols([mol_smiles])
            # print(f'Updating inventory: {mol_smiles}')
            self.inv = self.inv.append(mol_dict, ignore_index=True)
        # else:
            # print(f'{mol_smiles} already here')
        # del_len = len(self.inv) - orig_len
        # print(f'Change in inventory size: {del_len}')
        # print(f'Inventory size (updating): {len(self.inv)}')

    def TTC(self, time_H: float, time_M: float) -> float:
        """Calculate total time cost (TTC) for the reaction.

        Parameters
        ----------
        time_H: Human labor time (in h)
        time_M: Machine labor time (in h)

        Returns
        -------
        cost_t: Result of TTC calculation
        """
        cost_t: float = np.sqrt((self.a * time_H)**2 + (self.a_ * time_M)**2)
        # print('cost_t:', cost_t)
        return cost_t

    def money(self,
              reagents: List[dict],
              sm_costs: List[float],
              sm_eqs: List[float],
              time_H: float,
              time_M: float,
              rxn_scale: float,
              multiplier: int
              ) -> float:
        """Calculate the monetary cost of the reaction.

        Parameters
        ----------
        reagents:   List of dictionaries with reagent info
        sm_costs:   List of costs of starting materials
        sm_eqs:     List of equivalents for each starting material
        time_H:     Human labor time (in h)
        time_M:     Machine labor time (in h)
        rxn_scale:  Scale of the reaction (in mols)
        multiplier: Multiplier based on reactive sites calculated in 'stoichiometry'

        Returns
        -------
        cost_cad:   Total monetary cost of the reaction
        """
        rgt_cost: float = 0
        rgt_cost = sum([rgt['$/mol'] * rgt['eq'] * multiplier for rgt in reagents])
        rct_cost: float = 0
        rct_cost: float = sum([cost * eq for cost, eq in zip(sm_costs, sm_eqs)])
        # for cost, eq in zip(sm_costs, sm_eqs):
        #     if eq == 1:
        #         rct_cost += cost * eq
        #     else:
        #         rct_cost += cost * eq * multiplier
        mater_cost: float = rxn_scale * (rct_cost + rgt_cost)

        cost_cad: float = mater_cost + (time_H * self.C_H + time_M * self.C_M)
        return cost_cad

    def mass(self,
             reagents: List[dict],
             sm_MWs: List[float],
             sm_eqs: List[float],
             rxn_scale: float,
             multiplier: int
             ) -> float:
        """Calculate the mass cost of the reaction.

        Parameters
        ----------
        reagents:   List of dictionaries with reagent info
        sm_MWs:     List of molecular weights of starting materials
        sm_eqs:     List of equivalents for each starting material
        rxn_scale:  Scale of the reaction (in mols)
        multiplier: Multiplier based on reactive sites calculated in 'stoichiometry'

        Returns
        -------
        cost_mat:   Total mass cost of the reaction
        """
        rgt_quant: float = 0
        rgt_quant = sum([rgt['g/mol'] * rgt['eq'] * multiplier for rgt in reagents])
        rct_quant: float = 0
        rct_quant: float = sum([mw * eq for mw, eq in zip(sm_MWs, sm_eqs)])
        # for mw, eq in zip(sm_MWs, sm_eqs):
        #     if eq == 1:
        #         rct_quant += mw * eq
        #     else:
        #         rct_quant += mw * eq * multiplier
        cost_mat: float = rxn_scale * (rct_quant + rgt_quant)
        return cost_mat

    def similarity(self,
                   sm_smiles: str,
                   pdt_smiles: str,
                   tgt_smiles: str
                   ) -> float:
        """Calculate distance travelled by reaction using fingerprint similarities.

        Parameters
        ----------
        sm_smiles:  SMILES of the starting material molecule
        pdt_smiles: SMILES of the product molecule
        tgt_smiles: SMILES of the target molecule

        Returns
        -------
        travel: Distance that the reaction "travels" through chemical space
        """
        sim_BC = self.fp_similarity(pdt_smiles, tgt_smiles)
        sim_AC = self.fp_similarity(sm_smiles, tgt_smiles)

        if sim_BC < sim_AC:
            print('sim_BC:', sim_BC)
            print('sim_AC:', sim_AC)
            print('!!!\nATTN: Product less similar than SM!\n!!!')
            General().draw_mols([sm_smiles, pdt_smiles, tgt_smiles])

        if sim_AC == 0:
            sim_AC = 0.0000000001
        travel = sim_BC / (sim_AC)
        # TODO: Where sim_AC == 0, we need to avoid a divide by zero error.
        # if sim_AC == 0:
        #     sim_AC = 0.0000000001
        travel: float = sim_BC / (sim_AC + 0.0000000001)
        # if travel > 6:
        #     General().draw_mols([sm_smiles, pdt_smiles, tgt_smiles])
        return travel

    def fp_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate fingerprint simuilarity of 2 molecules.

        Parameters
        ----------
        smiles1: SMILES of the first molecule
        smiles2: SMILES of the second molecule

        Returns
        -------
        Fingerprint similarity of the two molecules
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        assert mol1 is not None, f'Invalid smiles {smiles1}'
        assert mol2 is not None, f'Invalid smiles {smiles2}'

        minP = 1
        maxP = 7
        fpSize = 8192

        fp1 = RDKFingerprint(mol1, minPath=minP, maxPath=maxP, fpSize=fpSize)
        fp2 = RDKFingerprint(mol2, minPath=minP, maxPath=maxP, fpSize=fpSize)

        return DataStructs.FingerprintSimilarity(fp1, fp2)

    def get_man_synth(self, sm_smiles: str, tgt_smiles: str, rxn_scale: float) -> Tuple[float, float, List[float]]:
        """Compare input to known list of 'manual' molecules. Return score, cost, reaction distances and number of steps.

        Parameters
        ----------
        sm_smiles: SMILES of the building block to be manually synthesized
        tgt_smiles: SMILES of the target molecule
        rxn_scale: Scale of the next automated reaction (i.e. quantity necessary to synthesize)

        Returns
        -------
        score:          StepScores for the manual syntheses
        man_molarCost:  Cost of the manual building block in $/mol
        l_man_dist:     List of distances for the manual reactions
        man_steps:      Number of manual reaction steps
        """
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
            man_yield: float = 1
            man_pdt_smiles: str = man_df[step]['pdt_smiles'][0]

            man_scale: float = rxn_scale / man_yield
            # man_t: float = 1 + np.sqrt(self.a * t_H**2 + self.a_ * t_M**2)
            man_time: float = self.TTC(t_H, t_M)
            # print('time cost:', man_t)
            man_money: float = ((self.C_H * t_H + self.C_M * t_M) + sum(man_df[step]['$/mol'] * man_df[step]['eq'] * man_scale))
            # print('$/mol:', man_d)
            man_materials: float = sum(man_df[step]['g/mol'] * man_df[step]['eq'] * man_scale)
            # print('mtrl cost:', man_m)
            cost: float = man_time * man_money * man_materials

            # manSM_df = man_df[step][man_df[step]['SM?'] == True]
            manSM_df = man_df[step][man_df[step]['SM?']]
            # SMs_list: List[str] = [row for row in manSM_df['SMILES']]
            # General().draw_mols(SMs_list)
            # General().draw_mols([man_pdt_smiles, tgt_smiles])

            sims_list: List[float] = [self.similarity(row, man_pdt_smiles, tgt_smiles) for row in manSM_df['SMILES']]
            # sims_list = []
            # for row in manSM_df['SMILES']:
            #     d = self.similarity(row, man_pdt_smiles, tgt_smiles)
            #     sims_list.append(d)
                # if d > 4:
                #     General().draw_mols([row, man_pdt_smiles, tgt_smiles])

            man_dist: float = min(sims_list)
            # if man_dist > 4:
            #     idx = sims_list.index(min(sims_list))
            #     really_efficient_mol = manSM_df['SMILES'][idx]
            #     General().draw_mols([really_efficient_mol, man_pdt_smiles, tgt_smiles])
            l_man_dist.append(man_dist)
            # print(l_man_dist)

            score += cost
            # man_molarCost = man_yld * man_money / man_scale
            man_molarCost = man_yield * man_money / man_scale
            # man_molarCost = 0
            # print('step cost:', cost)
            # print('step distance:', man_dist)
            # print('Man score:', score)

        man_steps = len(man_df)

        return score, man_molarCost, l_man_dist, man_steps

    def StepScore(self,
                  sm_list: List[dict],
                  product_smiles: str,
                  target_smiles: str,
                  rxn_type: str,
                  multiplier: int,
                  scale: float,
                  yld: float,
                  manual: bool,
                  ) -> float:
        """Perform calculations for the StepScore.

        Parameters
        ----------
        sm_list:        list of dictionaries corresponding to each starting material,
                        with SMILES and reaction equivalents
        product_smiles: SMILES of the desired product molecule
        target_smiles:  SMILES of the target molecule of the synthetic route
        rxn_type:       name of reaction to be carried out
        multiplier:     Multiplier (for reagents) based on number of reactive sites
        scale:          scale of the reaction in mols
        yld:            yield of the reaction
        manual:         whether the reaction if performed by a human (True) or a robot (False)

        Returns
        -------
        stepscore_results: dictionary containing relevant data for the stepscore
        stepscore_results = {
                             'StepScore': step_score,
                             'cost': cost,
                             'time': cost_time,
                             'money': cost_money,
                             'materials': cost_materials,
                             'yield': yld,
                             'distance': distance,
                             'man rxn distances': mandists_list,
                             '# man steps': man_steps
                             }
        """
        man_steps: float = 0
        man_stepscore: float = 0
        mandists_list: List[float] = []

        # General().draw_mols([sm['smiles'] for sm in sm_list])
        # print([sm['smiles'] for sm in sm_list])

        block_dicts: List[dict] = [self.get_block_info(sm['smiles']) for sm in sm_list]
        # print(block_dicts)

        reaction: list = General().load_pkl(TYPE_DIR, rxn_type)
        rxn_smry: dict = General().load_pkl(SMRY_DIR, f'{rxn_type}_summary')

        n_parr: int = rxn_smry['n_parr']  # Number of reactions performed in parallel
        t_H: float = rxn_smry['t_H'] / n_parr
        t_M: float = rxn_smry['t_M'] / n_parr
        # yld: float = rxn_smry['yield']
        # scale: float = rxn_smry['scale']

        # man_pre = ManualPreSynthesis()
        # lookup_df: pd.DataFrame = pd.read_excel(os.path.join(SMRY_DIR, 'ManSynth_lookup.xlsx'))

        man_blocks: List[dict] = [block for block in block_dicts if block['Manual?'] == 'Yes']
        # if len(man_blocks) > 0:
        #     print('Manual blocks:', man_blocks)

        # if any(molecule in man_blocks for molecule in lookup_df['SMILES']):
        for block in man_blocks:
            man_mol = {'score': 0, '$/mol': 0}
            # print('ATTN:    Manually synthesized starting material!')
            # General().draw_mols([block['SMILES']])
            man_mol['score'], man_Cmoney, mandists_list, man_steps = self.get_man_synth(block['SMILES'],
                                                                                        target_smiles,
                                                                                        scale)
            # print(f"Added {man_mol['score']} to StepScore.")
            # print(f"Cost of manual block is {block['$/mol']} $/mol.")
            man_stepscore += man_mol['score']

            # print(man_Cmoney)
            # block['$/mol'] = 0
            # print(block['$/mol'])

        sm_eqs: List[float] = [sm['eq'] for sm in sm_list]
        # eq_mult: int = self.stoichiometry(block_dicts, rxn_type)

        # Cost to travel through chemical space
        cost_time: float = self.TTC(t_H, t_M)
        # print('cost_time:', cost_time)
        block_costs: List[float] = [sm['$/mol'] for sm in block_dicts]
        cost_money: float = self.money(reaction,
                                       block_costs,
                                       sm_eqs,
                                       t_H,
                                       t_M,
                                       scale,
                                       multiplier)
        # t_money: float = t_H * self.C_H + t_M * self.C_M
        # m_money: float = cost_money - t_money
        # print('cost_money:', cost_money)
        block_MWs: List[float] = [sm['g/mol'] for sm in block_dicts]
        cost_materials: float = self.mass(reaction,
                                          block_MWs,
                                          sm_eqs,
                                          scale,
                                          multiplier)
        # print('cost_materials:', cost_materials)
        cost: float = cost_time * cost_money * cost_materials
        # print('cost:', cost)

        # Distance "traveled" in chemical space by reaction
        sims_list: List[float] = [self.similarity(sm['SMILES'], product_smiles, target_smiles) for sm in block_dicts]

        distance: float = min(sims_list)  # TODO: Rewrite to avoid the divide by zero error

        step_score: float = cost
        # print('StepScore w/o ManSynth:', step_score)
        step_score += man_stepscore
        # print('Cost:', cost)
        # print('Distance:', distance)
        # print('Manual StepScore:', man_stepscore)
        # print('StepScore:', step_score)

        MW: float = Descriptors.MolWt(Chem.MolFromSmiles(product_smiles))

        # molarCost = yld * cost_money / scale

        # product_dict: dict = {
        #     'Block_num': 0,
        #     'SMILES': product_smiles,
        #     'Name': '-',
        #     'g/mol': MW,
        #     'Quantity': 0,
        #     'CAD': 0,
        #     '$/mol': molarCost
        #     }

        if manual is True:
            man_steps += 1
        # molarCost = yld * cost_money / scale

        product_dict: dict = {
                              'Block_type': '-',
                              'Block_num': 0,
                              'SMILES': product_smiles,
                              'Name': '-',
                              'g/mol': MW,
                              'Quantity': 0,
                              'CAD': 0,
                              '$/mol': 0,
                              'Manual?': ''
                              }
        self.update_inventory(product_smiles, product_dict)

        self.inv.to_csv(inv_path, index=False)

        stepscore_results: dict = {
                                   'reaction': rxn_type,
                                   'StepScore': step_score,
                                   'cost': cost,
                                   'time': cost_time,
                                   'money': cost_money,
                                   'materials': cost_materials,
                                   'yield': yld,
                                   'distance': distance,
                                   'man rxn distances': mandists_list,
                                   '# man steps': man_steps
                                   }

        return stepscore_results

    def RouteScore(self, steps_list: List[float], total_yield: float) -> Tuple[float, int]:
        """Calculate the total RouteScore.

        Parameters
        ----------
        steps_list:  List of step details
        total_yield: Total yield (in mols) of the target molecule

        Returns
        -------
        route_score:     RouteScore for the route
        total_man_steps: Total number of manual reaction steps in the route
        """
        man_steps_l: List[int] = [step['# man steps'] for step in steps_list]
        total_man_steps: int = sum(man_steps_l)
        stepscores_list: List[float] = [step['StepScore'] for step in steps_list]
        distances_list: List[float] = [step['distance'] for step in steps_list]
        mandists_nested: List[List[float]] = [step['man rxn distances'] for step in steps_list]
        mandists_l: List[float] = [item for sublist in mandists_nested for item in sublist]
        all_distances: List[float] = distances_list + mandists_l

        sum_stepscores = sum(stepscores_list)
        cost_factor: float = sum_stepscores / total_yield
        maxDrxn = max(all_distances)
        avgDrxn = np.mean(all_distances)
        # TODO: How to account for longer routes having lower RS if they have large max Drxn?
        # num_steps = len(all_distances)
        # distance_factor: float = maxDrxn / (avgDrxn + num_steps)
        distance_factor: float = maxDrxn / avgDrxn

        # HACK: For now, simply removing distance from the route_score equation
        # route_score: float = cost_factor / distance_factor
        route_score: float = cost_factor

        routescore_results = {'RouteScore': route_score,
                              'Drxn factor': distance_factor,
                              'max Drxn': maxDrxn,
                              'avg Drxn': avgDrxn,
                              'Cost factor': cost_factor,
                              'sum Stepscores': sum_stepscores,
                              'n_Target': total_yield}
        return routescore_results, total_man_steps


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
