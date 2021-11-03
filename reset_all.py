import os
import glob
import pandas as pd
from typing import List


def rm_files(fldr: os.PathLike, filters_list: List[str], ext_list: List[str]):
    """Remove all files with specific filetype(s) from directory.

    Parameters
    ----------
    fldr:            Directory from which to remove files
    ext_list:       List of file types to remove files
    filters_list:   List of filters to use on filenames
    """
    for ext in ext_list:
        for filt in filters_list:
            for file in glob.glob(os.path.join(fldr, f'{filt}.{ext}')):
                try:
                    os.remove(file)
                except FileNotFoundError:
                    print(f'{filt}.{ext} does not exist.')


# Define directories
HERE = os.path.abspath(os.path.dirname(__file__))
INV_DIR = os.path.join(HERE, 'Inventory')
PROP_DIR = os.path.join(HERE, 'Properties')
RSLT_DIR = os.path.join(HERE, 'Results')
TGT_DIR = os.path.join(HERE, 'Targets')

# Reset 'Inventory.csv' file to clean state
inv = pd.read_csv(os.path.join(INV_DIR, 'Inventory.csv'))
inv = inv[~inv.Block_type.isin(['-'])]
inv.to_csv(os.path.join(INV_DIR, 'Inventory.csv'))
print('Reset Inventory.csv')

# Remove 'full_props' file
rm_files(PROP_DIR, ['full_props'], ['pkl'])
print('Removed full_props.pkl')

# Reset 'Results' directory
rm_files(RSLT_DIR, ['*'], ['pkl'])
print('Reset Results folder.')

# Reset 'Targets' directory
rm_files(TGT_DIR, ['targets_*'], ['pkl', 'csv'])
print('Reset Targets folder.')
