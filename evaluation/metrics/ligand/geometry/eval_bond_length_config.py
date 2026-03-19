import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Default path for CCD bond length distribution
DEFAULT_CCD_BOND_LENGTH_PATH = os.path.join(current_dir, 'ccd_bond_length_distribution.npy')# Global variable to store the distribution path
_ccd_bond_length_path = DEFAULT_CCD_BOND_LENGTH_PATH

def set_ccd_bond_length_path(path):
    """Set the path for CCD bond length distribution file"""
    global _ccd_bond_length_path
    _ccd_bond_length_path = path

def get_ccd_bond_length_path():
    """Get the current path for CCD bond length distribution file"""
    return _ccd_bond_length_path

BOND_TYPES = frozenset(((6, 6, 1), (6, 6, 2), (6, 6, 4), (6, 7, 1), (6, 7, 2), (6, 7, 4), (6, 8, 1), (6, 8, 2),))

DISTANCE_BINS = np.arange(1.1, 1.7, 0.005)[:-1]

EMPIRICAL_BINS = {
    'CC_2A': np.linspace(0, 2, 100),
    'All_12A': np.linspace(0, 12, 100)
}

# Load distributions
EMPIRICAL_DISTRIBUTIONS = np.load(_ccd_bond_length_path, allow_pickle=True).tolist()
EMPIRICAL_DISTRIBUTIONS = {k: v for k, v in EMPIRICAL_DISTRIBUTIONS.items() if k in BOND_TYPES}
# breakpoint()
# assert set(BOND_TYPES) == set(EMPIRICAL_DISTRIBUTIONS.keys())

for v in EMPIRICAL_DISTRIBUTIONS.values():
    assert len(DISTANCE_BINS) + 1 == len(v)
