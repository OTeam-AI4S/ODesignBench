"""Utils for evaluating bond length."""

import collections
from typing import Tuple, Sequence, Dict, Optional

import numpy as np
from scipy import spatial as sci_spatial
import matplotlib.pyplot as plt

from .eval_bond_length_config import DISTANCE_BINS, EMPIRICAL_DISTRIBUTIONS
from metrics.ligand.utils.molecule.constants import *

BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
BondLengthData = Tuple[BondType, float]  # (bond_type, bond_length)
BondLengthProfile = Dict[BondType, np.ndarray]  # bond_type -> empirical distribution
def get_bond_str(bond_type):
    if bond_type == 1:
        return '-'
    elif bond_type == 2:
        return '='
    elif bond_type == 3:
        return '#'
    else:
        return '?'

def get_distribution(distances: Sequence[float], bins=DISTANCE_BINS) -> np.ndarray:
    """Get the distribution of distances.

    Args:
        distances (list): List of distances.
        bins (list): bins of distances
    Returns:
        np.array: empirical distribution of distances with length equals to DISTANCE_BINS.
    """
    bin_counts = collections.Counter(np.searchsorted(bins, distances))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)
    return bin_counts


def _format_bond_type(bond_type: BondType) -> BondType:
    atom1, atom2, bond_category = bond_type
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1
    return atom1, atom2, bond_category


def get_bond_length_profile(bond_lengths: Sequence[BondLengthData]) -> BondLengthProfile:
    bond_length_profile = collections.defaultdict(list)
    for bond_type, bond_length in bond_lengths:
        bond_type = _format_bond_type(bond_type)
        bond_length_profile[bond_type].append(bond_length)
    bond_length_profile = {k: get_distribution(v) for k, v in bond_length_profile.items()}
    return bond_length_profile


def _bond_type_str(bond_type: BondType) -> str:
    atom1, atom2, bond_category = bond_type
    return f'{atom1}{get_bond_str(bond_category)}{atom2}'


def eval_bond_length_profile(bond_length_profile: BondLengthProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for bond_type, gt_distribution in EMPIRICAL_DISTRIBUTIONS.items():
        if bond_type not in bond_length_profile:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = None
        else:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(gt_distribution,bond_length_profile[bond_type])
    return metrics

def bond_distance_from_mol(mol):
    pos = mol.GetConformer().GetPositions()
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    all_distances = []
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_types[bond.GetBondType()]
        distance = pdist[s_idx, e_idx]
        all_distances.append(((s_sym, e_sym, bond_type), distance))
    return all_distances