import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CCD_BOND_ANGLE_PATH = os.path.join(current_dir, 'ccd_bond_angle_distribution.npy')

# Global variable to store the distribution path
_ccd_bond_angle_path = DEFAULT_CCD_BOND_ANGLE_PATH

def set_ccd_bond_angle_path(path):
    """Set the path for CCD bond angle distribution file"""
    global _ccd_bond_angle_path
    _ccd_bond_angle_path = path

def get_ccd_bond_angle_path():
    """Get the current path for CCD bond angle distribution file"""
    return _ccd_bond_angle_path

# Load distributions
EMPIRICAL_DISTRIBUTIONS = np.load(_ccd_bond_angle_path, allow_pickle=True).tolist()