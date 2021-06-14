import os
import pickle
from typing import Tuple, List
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors, Draw
from IPython.display import display
IPythonConsole.ipython_useSVG = True

# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
SMRY_DIR = os.path.join(HERE, 'Rxn_Summs')
TYPE_DIR = os.path.join(HERE, 'Rxn_Type')
INV_DIR = os.path.join(HERE, 'Inventory')
PROP_DIR = os.path.join(HERE, 'Properties')

# Paths to properties and inventory files
inv_path = os.path.join(INV_DIR, 'Inventory.csv')


class General:
    """Class for general functions and data management operations."""

    def CustomError(self, func, message: str):
        """Raise a custom error and print message using assert."""
        print(message)
        return func

    def load_pkl(self, folder: str, file: str):
        """Load a .pkl file.

        Parameters
        ----------
        folder:     Directory where the file is located. Must be within HERE.
        file:       Filename of the .pkl file

        Returns
        -------
        pkl: Contents of the .pkl file
        """
        load_path = os.path.join(HERE, folder, f'{file}.pkl')
        pkl = pickle.load(open(load_path, 'rb'))
        return pkl

    def draw_mols(self, smiles_list: List[str]):
        """Draw molecular structures inline.

        Parameters
        ----------
        smiles_list: List of SMILES to draw

        Returns
        -------
        Nothing
        """
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
        df: Appropriately formatted dataframe
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
                steps_list: List[dict],
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
        # Mass of target molecule obtained
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
        # Dictionary for pattern matching and counting reaction sites
        # Keys correspond to the type of reaction
        # Values correspond to the type of substructure to match
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
        mult: int = sum(matches_list)
        # If no reaction sites are found, show AssertionError and display the molecule
        assert mult != 0, General().CustomError(General().draw_mols([smiles]), 'mult = 0!')
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
        # List of reactant molecules and eq per reaction site
        sm_list: List[dict] = [{'smiles': smilesA, 'eq': 3},
                               {'smiles': smilesB, 'eq': 1}
                               ]
        # Number of reaction sites on the B molecule (1)
        rxn_sites = self.stoichiometry(smilesB, 'Suzuki')
        # Total equivalents of A molecule
        sm_list[0]['eq'] = rxn_sites * sm_list[0]['eq']
        # Unlike subsequent reactions, scale isn't normalized because for the limiting reagent
        # We treat commercially available materials as being available in essentially unlimited quantities
        wS_scale: float = scale
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
        # List of reactant molecules and eq per reaction site
        sm_list: List[dict] = [
                               {'smiles': smilesAB, 'eq': 3},
                               {'smiles': smilesC, 'eq': 1}
                               ]
        # Number of reaction sites on the C molecule (2)
        rxn_sites = self.stoichiometry(smilesC, 'Suzuki')
        # Total equivalents of AB molecule
        sm_list[0]['eq'] = rxn_sites * sm_list[0]['eq']
        # Normalize scale for quantity of limiting reagent (AB)
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
        # List of reactant molecules and eq per reaction site
        sm_list: List[dict] = [
                               {'smiles': smilesNBoc, 'eq': 1}
                               ]
        # Number of reaction sites for Boc-deprotection (2)
        rxn_sites = self.stoichiometry(smilesNBoc, 'deBoc')
        # Normalize scale for quantity of limiting reagent
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
        # List of reactant molecules and eq per reaction site
        sm_list: List[dict] = [
                               {'smiles': smilesNH, 'eq': 1},
                               {'smiles': smilesX, 'eq': 3}
                               ]
        # Number of reaction sites for BHA reaction (2)
        rxn_sites = self.stoichiometry(smilesBHA, 'BHA-Py')
        # Total equivalents of halide
        sm_list[1]['eq'] = rxn_sites * sm_list[1]['eq']
        # Normalize scale for quantity of limiting reagent
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
        # List of reactant molecules and eq per reaction site
        sm_list: List[dict] = [
                               {'smiles': smilesAr, 'eq': 1},
                               {'smiles': smilesNu, 'eq': 2}
                               ]
        # Number of reaction sites for SNAr reaction (2)
        rxn_sites = self.stoichiometry(smilesSNAr, 'SNAr-Cz')
        # Total equivalents of carbazole
        sm_list[1]['eq'] = rxn_sites * sm_list[1]['eq']
        # Normalize scale for quantity of limiting reagent
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

    # Load values of C_H and C_M terms
    rates_path = os.path.join(SMRY_DIR, 'hrly_costs.pkl')
    hrly_rates = pickle.load(open(rates_path, 'rb'))
    C_H: float = hrly_rates['C_H']
    C_M: float = hrly_rates['C_M']
    a: float = C_H / C_M
    a_: float = 1 / a

    def __init__(self):
        # Load inventory
        self.inv = None
        self.inv = pd.read_csv(inv_path)

    def get_block_info(self, smilesBlock: str) -> dict:
        """Get block information dict from inventory dataframe.

        Parameters
        ----------
        smilesBlock: SMILES of the building block to search for in inventory

        Returns
        -------
        block_info: Dictionary containing inventory information for smilesBlock molecule
        """
        block_entry = self.inv[self.inv['SMILES'] == smilesBlock]
        block_info: dict = block_entry.to_dict(orient='records')[0]

        return block_info

    def update_inventory(self, mol_smiles: str, mol_dict: dict):
        """Add new molecule to inventory if not already present.

        Parameters
        ----------
        mol_smiles: SMILES of molecule to add to inventory
        mol_dict:   Dictionary of new molecule's inventory data

        Returns
        -------
        Nothing
        """
        if mol_smiles not in self.inv.SMILES.values:
            self.inv = self.inv.append(mol_dict, ignore_index=True)

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
        # Calculate total cost of reagents
        rgt_cost: float = 0
        rgt_cost = sum([rgt['$/mol'] * rgt['eq'] * multiplier for rgt in reagents])
        # Calculate total cost of reactants
        rct_cost: float = 0
        rct_cost: float = sum([cost * eq for cost, eq in zip(sm_costs, sm_eqs)])
        # Factor cost based on reaction scale
        mater_cost: float = rxn_scale * (rct_cost + rgt_cost)
        # Add labor cost
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
        multiplier: Multiplier based on reaction sites calculated in 'stoichiometry'

        Returns
        -------
        cost_mat:   Total mass cost of the reaction
        """
        # Calculate total mass of reagents required
        rgt_quant: float = 0
        rgt_quant = sum([rgt['g/mol'] * rgt['eq'] * multiplier for rgt in reagents])
        # Calculate total mass of reactants required
        rct_quant: float = 0
        rct_quant: float = sum([mw * eq for mw, eq in zip(sm_MWs, sm_eqs)])
        # Factor mass required based on reaction scale
        cost_mat: float = rxn_scale * (rct_quant + rgt_quant)
        return cost_mat

    def get_man_synth(self, sm_smiles: str, tgt_smiles: str, rxn_scale: float) -> Tuple[float, float, int]:
        """Compare input to known list of 'manual' molecules. Return score, cost and number of steps.

        Parameters
        ----------
        sm_smiles: SMILES of the building block to be manually synthesized
        tgt_smiles: SMILES of the target molecule
        rxn_scale: Scale of the next automated reaction (i.e. quantity necessary to synthesize)

        Returns
        -------
        score:          StepScores for the manual syntheses
        man_molarCost:  Cost of the manual building block in $/mol
        man_steps:      Number of manual reaction steps
        """
        # Load dataframe of manual syntheses
        lookup_df = pd.DataFrame()
        lookup_df = pd.read_excel(os.path.join(SMRY_DIR, 'ManSynth_lookup.xlsx'))
        smry_fname: str = lookup_df['summary file'][lookup_df['SMILES'] == sm_smiles].iloc[0]
        man_df: dict = pd.read_excel(os.path.join(SMRY_DIR, smry_fname), sheet_name=None, dtype={'SM?': bool})

        # For each step in manual synthesis, calculate StepScore
        score: float = 0
        for step in man_df:
            t_H: float = man_df[step]['t_H'][0]
            t_M: float = man_df[step]['t_M'][0]
            man_yield: float = 1
            man_scale: float = rxn_scale / man_yield
            man_time: float = self.TTC(t_H, t_M)
            man_money: float = ((self.C_H * t_H + self.C_M * t_M) + sum(man_df[step]['$/mol'] * man_df[step]['eq'] * man_scale))
            man_materials: float = sum(man_df[step]['g/mol'] * man_df[step]['eq'] * man_scale)
            cost: float = man_time * man_money * man_materials

            score += cost
            man_molarCost = man_yield * man_money / man_scale

        man_steps = len(man_df)
        return score, man_molarCost, man_steps

    def StepScore(self,
                  sm_list: List[dict],
                  product_smiles: str,
                  target_smiles: str,
                  rxn_type: str,
                  multiplier: int,
                  scale: float,
                  yld: float,
                  manual: bool,
                  ) -> dict:
        """Perform calculations for the StepScore.

        Parameters
        ----------
        sm_list:        list of dictionaries corresponding to each starting material,
                        with SMILES and reaction equivalents
        product_smiles: SMILES of the desired product molecule
        target_smiles:  SMILES of the target molecule of the synthetic route
        rxn_type:       name of reaction to be carried out
        multiplier:     Multiplier (for reagents) based on number of reaction sites
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
                             '# man steps': man_steps
                             }
        """
        man_steps: float = 0
        man_stepscore: float = 0

        # Get information on starting materials from inventory
        block_dicts: List[dict] = [self.get_block_info(sm['smiles']) for sm in sm_list]
        # Get information on reagents for the reaction
        reaction: list = General().load_pkl(TYPE_DIR, rxn_type)
        # Get reaction summary information
        rxn_smry: dict = General().load_pkl(SMRY_DIR, f'{rxn_type}_summary')

        n_parr: int = rxn_smry['n_parr']  # Number of reactions performed in parallel
        t_H: float = rxn_smry['t_H'] / n_parr
        t_M: float = rxn_smry['t_M'] / n_parr

        # If there are manually synthesized blocks in the route, calculate cost of manual synthesis
        man_blocks: List[dict] = [block for block in block_dicts if block['Manual?'] == 'Yes']
        for block in man_blocks:
            man_mol = {'score': 0, '$/mol': 0}
            # StepScore, monetary cost and # of steps in manual synthesis
            man_mol['score'], man_Cmoney, man_steps = self.get_man_synth(block['SMILES'],
                                                                         target_smiles,
                                                                         scale)
            man_stepscore += man_mol['score']

        # List of equivalents for each reactant
        sm_eqs: List[float] = [sm['eq'] for sm in sm_list]

        # Calculate costs of synthesis
        # Time cost
        cost_time: float = self.TTC(t_H, t_M)
        block_costs: List[float] = [sm['$/mol'] for sm in block_dicts]
        # Monetary cost
        cost_money: float = self.money(reaction,
                                       block_costs,
                                       sm_eqs,
                                       t_H,
                                       t_M,
                                       scale,
                                       multiplier)
        block_MWs: List[float] = [sm['g/mol'] for sm in block_dicts]
        # Materials cost
        cost_materials: float = self.mass(reaction,
                                          block_MWs,
                                          sm_eqs,
                                          scale,
                                          multiplier)
        cost: float = cost_time * cost_money * cost_materials

        step_score: float = cost
        step_score += man_stepscore

        MW: float = Descriptors.MolWt(Chem.MolFromSmiles(product_smiles))

        if manual is True:
            man_steps += 1

        # Add product information to inventory
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
                                   '# man steps': man_steps
                                   }

        return stepscore_results

    def RouteScore(self, steps_list: List[dict], total_yield: float) -> Tuple[dict, int]:
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
        # List containing # of all manual synthesis steps
        man_steps_l: List[int] = [step['# man steps'] for step in steps_list]
        # Total number of manual synthesis steps
        total_man_steps: int = sum(man_steps_l)
        # List of all StepScores
        stepscores_list: List[float] = [step['StepScore'] for step in steps_list]
        # Sum of all StepScores
        sum_stepscores = sum(stepscores_list)
        # Calculate RouteScore
        cost_factor: float = sum_stepscores / total_yield
        route_score: float = cost_factor
        # print(route_score)

        routescore_results = {'RouteScore': route_score,
                              'Cost factor': cost_factor,
                              'sum Stepscores': sum_stepscores,
                              'n_Target': total_yield}
        return routescore_results, total_man_steps
