from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem, TorsionFingerprints
# from .eval_bond_angle_config import *
from scipy import spatial as sci_spatial
from metrics.ligand.utils.molecule.constants import *
from tqdm.auto import tqdm
import pickle
from metrics.ligand.geometry.eval_torsion_angle_config import EMPIRICAL_DISTRIBUTIONS

from itertools import combinations
from typing import Tuple, Sequence, Dict, Optional
import numpy as np
import collections
import os

TorsionType = Tuple[int, int, int, int, int, int, int]  # (atomic_num, bond_type, atomic_num, bond_type, atomic_num, bond_type, atomic_num)

def get_bond_type(bond):
    return bond_types[bond.GetBondType()]

def get_bond_str(bond_type):
    if bond_type == 1:
        return '-'
    elif bond_type == 2:
        return '='
    elif bond_type == 3:
        return '#'
    else:
        return '?'

TorsionAngleData = Tuple[TorsionType, float] 

def get_distribution(angles: Sequence[float], bins=np.arange(-180, 180, 2)) -> np.ndarray:

    bin_counts = collections.Counter(np.searchsorted(bins, angles))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)
    return bin_counts

TorsionAngleProfile = Dict[TorsionType, np.ndarray] 

def get_torsion_angle_profile(torsion_angles: Sequence[TorsionAngleData]) -> TorsionAngleProfile:
    torsion_angle_profile = collections.defaultdict(list)
    for torsion_type, torsion_angle in torsion_angles:
        torsion_angle_profile[torsion_type].append(torsion_angle)
    torsion_angle_profile = {k: get_distribution(v) for k, v in torsion_angle_profile.items()}
    return torsion_angle_profile

def _torsion_type_str(torsion_type: TorsionType) -> str:
    atom1, bond12, atom2, bond23, atom3, bond34, atom4 = torsion_type
    return f'{atom1}{get_bond_str(bond12)}{atom2}{get_bond_str(bond23)}{atom3}{get_bond_str(bond34)}{atom4}'

def eval_torsion_angle_profile(torsion_angle_profile: TorsionAngleProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for torsion_type, gt_distribution in EMPIRICAL_DISTRIBUTIONS.items():
        if torsion_type not in torsion_angle_profile:
            metrics[f'JSD_{_torsion_type_str(torsion_type)}'] = None
        else:
            metrics[f'JSD_{_torsion_type_str(torsion_type)}'] = sci_spatial.distance.jensenshannon(gt_distribution,torsion_angle_profile[torsion_type])

    return metrics
def torsion_angle_from_mol(mol):
    angles = []
    types = []
    conf = mol.GetConformer(id=0)
    
    # Find all torsion angles (dihedral angles) in the molecule
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        
        # Get neighbors of atom1 (excluding atom2)
        atom1_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom1_idx).GetNeighbors() if n.GetIdx() != atom2_idx]
        # Get neighbors of atom2 (excluding atom1)
        atom2_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom2_idx).GetNeighbors() if n.GetIdx() != atom1_idx]
        
        # Create torsion angles: atom0-atom1-atom2-atom3
        for atom0_idx in atom1_neighbors:
            for atom3_idx in atom2_neighbors:
                try:
                    # Calculate torsion angle
                    torsion_angle = rdMolTransforms.GetDihedralDeg(conf, atom0_idx, atom1_idx, atom2_idx, atom3_idx)
                    
                    # Create torsion type tuple
                    torsion_type = (
                        mol.GetAtomWithIdx(atom0_idx).GetAtomicNum(),
                        get_bond_type(mol.GetBondBetweenAtoms(atom0_idx, atom1_idx)),
                        mol.GetAtomWithIdx(atom1_idx).GetAtomicNum(),
                        get_bond_type(mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)),
                        mol.GetAtomWithIdx(atom2_idx).GetAtomicNum(),
                        get_bond_type(mol.GetBondBetweenAtoms(atom2_idx, atom3_idx)),
                        mol.GetAtomWithIdx(atom3_idx).GetAtomicNum()
                    )
                    
                    angles.append(torsion_angle)
                    types.append(torsion_type)
                except:
                    # Skip if angle calculation fails
                    continue

    return [(torsion_type, angle) for torsion_type, angle in zip(types, angles)]
