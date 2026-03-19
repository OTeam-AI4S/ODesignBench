import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Default path for CCD torsion angle distribution
DEFAULT_CCD_TORSION_ANGLE_PATH = os.path.join(current_dir, 'ccd_torsion_angle_distribution.npy')
# Global variable to store the distribution path
_ccd_torsion_angle_path = DEFAULT_CCD_TORSION_ANGLE_PATH

def set_ccd_torsion_angle_path(path):
    """Set the path for CCD torsion angle distribution file"""
    global _ccd_torsion_angle_path
    _ccd_torsion_angle_path = path

def get_ccd_torsion_angle_path():
    """Get the current path for CCD torsion angle distribution file"""
    return _ccd_torsion_angle_path


# Load empirical distributions for torsion angles
EMPIRICAL_DISTRIBUTIONS = np.load(_ccd_torsion_angle_path, allow_pickle=True).tolist()