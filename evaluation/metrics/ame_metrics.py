"""
AME (Atomic Motif Enzyme) specific metrics.

Implements metrics from RFdiffusion2 AME benchmark:
- catalytic_constraints: 6 criteria for catalytic constraint satisfaction
- default: clash detection metrics
- backbone: ligand distance and secondary structure metrics
- sidechain: sidechain RMSD metrics

Note: All refold-related metrics use chai-1 results (not AF2).
"""

import pickle
import numpy as np
import scipy.spatial.distance
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from functools import partial
import warnings

import biotite.structure as struc
from biotite.structure.io import pdb, pdbx
from biotite.structure import AtomArray

from evaluation.metrics.rmsd import RMSDCalculator


def _read_structure(path: str | Path) -> AtomArray:
    """Read structure from PDB or CIF file."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".cif":
        cif = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(cif, model=1)
    if suf == ".pdb":
        pdb_file = pdb.PDBFile.read(str(path))
        return pdb_file.get_structure(model=1)
    raise ValueError(f"Unsupported structure format: {path}")


def _get_aligner(f: np.ndarray, t: np.ndarray) -> callable:
    """
    Get alignment function that aligns f to t using Kabsch algorithm.
    
    Args:
        f: Source coordinates [N, 3]
        t: Target coordinates [N, 3]
        
    Returns:
        Function that takes coordinates [M, 3] and returns aligned coordinates
    """
    if len(f) != len(t):
        raise ValueError(f"Coordinate arrays must have same length: {len(f)} vs {len(t)}")
    if len(f) < 3:
        # Not enough points for alignment, return identity
        def align_func(coords: np.ndarray) -> np.ndarray:
            return coords
        return align_func
    
    # Center both sets
    f_centered = f - f.mean(axis=0)
    t_centered = t - t.mean(axis=0)
    
    # Compute rotation matrix using SVD (Kabsch algorithm)
    H = f_centered.T @ t_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure right-handed rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    f_mean = f.mean(axis=0)
    t_mean = t.mean(axis=0)
    
    def align_func(coords: np.ndarray) -> np.ndarray:
        """Align coordinates using computed rotation and translation."""
        return (coords - f_mean) @ R.T + t_mean
    
    return align_func


def _rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate arrays."""
    if len(coords1) != len(coords2):
        raise ValueError(f"Coordinate arrays must have same length: {len(coords1)} vs {len(coords2)}")
    return float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))


def _parse_motif_residues(motif_str: str) -> List[str]:
    """Parse motif residues string into list."""
    if not motif_str or str(motif_str).strip() == '':
        return []
    residues = [r.strip() for r in str(motif_str).split(',')]
    return [r for r in residues if r]


def _get_motif_atoms(
    structure: AtomArray,
    motif_residues: List[str],
    atom_names: Optional[List[str]] = None,
) -> AtomArray:
    """
    Extract motif atoms from structure.
    
    Args:
        structure: Structure atom array
        motif_residues: List of residue identifiers like ["A22", "A48"]
        atom_names: Optional list of atom names to filter (e.g., ["N", "CA", "C"])
                    If None, returns all heavy atoms
        
    Returns:
        AtomArray containing motif atoms
    """
    # Parse motif residues into (chain_id, res_id) tuples
    motif_set = set()
    for res_str in motif_residues:
        # Parse format like "A22" -> (chain="A", res_id=22)
        if len(res_str) < 2:
            continue
        chain_id = res_str[0]
        try:
            res_id = int(res_str[1:])
            motif_set.add((chain_id, res_id))
        except ValueError:
            continue
    
    if not motif_set:
        return structure[[]]  # Empty array
    
    # Filter structure to motif residues
    mask = np.zeros(len(structure), dtype=bool)
    for i, (chain_id, res_id) in enumerate(zip(structure.chain_id, structure.res_id)):
        # Handle res_id format (could be int or tuple)
        if isinstance(res_id, tuple):
            pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
        else:
            pdb_res_id = res_id
        
        if (str(chain_id), int(pdb_res_id)) in motif_set:
            mask[i] = True
    
    motif_atoms = structure[mask]
    
    # Filter by atom names if specified
    if atom_names is not None:
        atom_mask = np.isin(motif_atoms.atom_name, atom_names)
        motif_atoms = motif_atoms[atom_mask]
    
    # Filter to heavy atoms only
    heavy_mask = motif_atoms.element != "H"
    motif_atoms = motif_atoms[heavy_mask]
    
    return motif_atoms


def _get_ligand_atoms(structure: AtomArray) -> AtomArray:
    """Extract ligand atoms (HETATM) from structure."""
    ligand_mask = structure.hetero & (structure.element != "H")
    return structure[ligand_mask]


def _get_backbone_atoms(structure: AtomArray) -> AtomArray:
    """Extract backbone atoms (N, CA, C, O) from structure."""
    bb_mask = (
        np.isin(structure.atom_name, ["N", "CA", "C", "O"])
        & (~structure.hetero)
    )
    return structure[bb_mask]


def _get_ca_atoms(structure: AtomArray) -> AtomArray:
    """Extract CA atoms from structure."""
    ca_mask = (structure.atom_name == "CA") & (~structure.hetero)
    return structure[ca_mask]


def compute_catalytic_constraints(
    ref_pdb: str | Path,
    des_pdb: str | Path,
    chai_pdb: str | Path,
    motif_residues: List[str],
    chai_plddt: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute catalytic_constraints metrics (6 criteria).
    
    Args:
        ref_pdb: Reference structure (input PDB)
        des_pdb: Design structure (backbone from diffusion/model)
        chai_pdb: Chai-1 refold structure
        motif_residues: List of motif residue identifiers like ["A22", "A48"]
        chai_plddt: Chai-1 plDDT value (if available)
        
    Returns:
        Dictionary with metric values:
        - criterion_1_metric: Motif+Ligand RMSD (design vs ref, motif-aligned)
        - criterion_1: Boolean (pass if < 1.0 Å)
        - criterion_2_metric: Motif RMSD (chai vs design, motif-aligned)
        - criterion_2: Boolean (pass if < 1.0 Å)
        - criterion_3_metric: CA RMSD (chai vs design, CA-aligned)
        - criterion_3: Boolean (pass if < 1.5 Å)
        - criterion_4_metric: Min backbone-ligand distance (design)
        - criterion_4: Boolean (pass if all > 2.0 Å)
        - criterion_5_metric: Min backbone-ligand distance (chai)
        - criterion_5: Boolean (pass if all > 2.0 Å)
        - criterion_6_metric: Chai-1 plDDT
        - criterion_6: Boolean (pass if > 0.7)
        - catalytic_heavy_atom_rmsd: Heavy atom RMSD of catalytic residues (chai vs design, aligned to chai motif backbone N, CA, C)
        - ligand_clash_1_5A_metric: Min backbone-ligand distance (chai) with 1.5 Å threshold
        - ligand_clash_1_5A: Boolean (pass if min distance > 1.5 Å)
        - rfd2_success: Boolean (pass if catalytic_heavy_atom_rmsd < 1.5 Å and ligand_clash_1_5A passes)
    """
    out = {}
    
    # Read structures
    ref_struct = _read_structure(ref_pdb)
    des_struct = _read_structure(des_pdb)
    chai_struct = _read_structure(chai_pdb)
    
    # Get motif atoms (all heavy atoms)
    ref_motif = _get_motif_atoms(ref_struct, motif_residues)
    des_motif = _get_motif_atoms(des_struct, motif_residues)
    chai_motif = _get_motif_atoms(chai_struct, motif_residues)
    
    # Helper function to match atoms by name (defined here so it can be reused)
    def match_atoms_by_name(atoms1, atoms2):
        """Match atoms by (chain_id, res_id, atom_name)."""
        coords1 = []
        coords2 = []
        # Create mapping for atoms1
        map1 = {}
        for i, (chain_id, res_id, atom_name) in enumerate(
            zip(atoms1.chain_id, atoms1.res_id, atoms1.atom_name)
        ):
            if isinstance(res_id, tuple):
                pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
            else:
                pdb_res_id = res_id
            key = (str(chain_id), int(pdb_res_id), str(atom_name))
            if key not in map1:  # Keep first occurrence
                map1[key] = atoms1.coord[i]
        
        # Match atoms2 to atoms1
        for i, (chain_id, res_id, atom_name) in enumerate(
            zip(atoms2.chain_id, atoms2.res_id, atoms2.atom_name)
        ):
            if isinstance(res_id, tuple):
                pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
            else:
                pdb_res_id = res_id
            key = (str(chain_id), int(pdb_res_id), str(atom_name))
            if key in map1:
                coords1.append(map1[key])
                coords2.append(atoms2.coord[i])
        
        return np.array(coords1), np.array(coords2)
    
    if len(ref_motif) == 0 or len(des_motif) == 0:
        warnings.warn(f"No motif atoms found. Returning NaN metrics.")
        return {
            'criterion_1_metric': np.nan,
            'criterion_1': False,
            'criterion_2_metric': np.nan,
            'criterion_2': False,
            'criterion_3_metric': np.nan,
            'criterion_3': False,
            'criterion_4_metric': np.nan,
            'criterion_4': False,
            'criterion_5_metric': np.nan,
            'criterion_5': False,
            'criterion_6_metric': np.nan if chai_plddt is None else chai_plddt,
            'criterion_6': False if chai_plddt is None else (chai_plddt > 0.7),
            'catalytic_heavy_atom_rmsd': np.nan,
            'ligand_clash_1_5A_metric': np.nan,
            'ligand_clash_1_5A': False,
            'ligand_clash_count_1_5A': 0,
            'rfd2_success': False,
        }
    
    # Criterion 1: Motif+Ligand RMSD (design vs ref, motif-aligned)
    ref_ligand = _get_ligand_atoms(ref_struct)
    des_ligand = _get_ligand_atoms(des_struct)
    
    if len(ref_ligand) > 0 and len(des_ligand) > 0:
        # Match motif atoms first
        ref_motif_coords, des_motif_coords = match_atoms_by_name(ref_motif, des_motif)
        
        # Check if we have matching atoms for alignment
        if len(ref_motif_coords) == 0 or len(des_motif_coords) == 0:
            criterion_1_rmsd = np.nan
        elif len(ref_motif_coords) != len(des_motif_coords):
            criterion_1_rmsd = np.nan
        else:
            # Align design motif to reference motif
            aligner = _get_aligner(des_motif_coords, ref_motif_coords)
            
            # Align motif + ligand
            des_motif_aligned = aligner(des_motif_coords)
            des_ligand_aligned = aligner(des_ligand.coord)
        
        # Match ligand atoms by atom_name (ligands may have same atom names)
        ref_ligand_names = ref_ligand.atom_name.tolist()
        des_ligand_names = des_ligand.atom_name.tolist()
        common_ligand_names = sorted(set(ref_ligand_names) & set(des_ligand_names))
        
        ref_ligand_coords = []
        des_ligand_coords = []
        for name in common_ligand_names:
            ref_idx = ref_ligand_names.index(name)
            des_idx = des_ligand_names.index(name)
            ref_ligand_coords.append(ref_ligand.coord[ref_idx])
            des_ligand_coords.append(des_ligand_aligned[des_idx])
        
        # Combine motif and ligand
        if len(ref_motif_coords) > 0 and len(ref_ligand_coords) > 0:
            motif_and_ligand_ref = np.vstack([ref_motif_coords, np.array(ref_ligand_coords)])
            motif_and_ligand_des = np.vstack([des_motif_coords, np.array(des_ligand_coords)])
            criterion_1_rmsd = _rmsd(motif_and_ligand_ref, motif_and_ligand_des)
        elif len(ref_motif_coords) > 0:
                criterion_1_rmsd = _rmsd(ref_motif_coords, des_motif_aligned)
        else:
            criterion_1_rmsd = np.nan
    else:
        # No ligand, just motif RMSD - match atoms first
        ref_motif_coords, des_motif_coords = match_atoms_by_name(ref_motif, des_motif)
        
        if len(ref_motif_coords) == 0 or len(des_motif_coords) == 0:
            criterion_1_rmsd = np.nan
        elif len(ref_motif_coords) != len(des_motif_coords):
            criterion_1_rmsd = np.nan
        else:
            aligner = _get_aligner(des_motif_coords, ref_motif_coords)
            des_motif_aligned = aligner(des_motif_coords)
            criterion_1_rmsd = _rmsd(ref_motif_coords, des_motif_aligned)
    
    out['criterion_1_metric'] = criterion_1_rmsd
    out['criterion_1'] = criterion_1_rmsd < 1.0
    
    # Criterion 2: Motif RMSD (chai vs design, motif-aligned)
    if len(chai_motif) > 0:
        # Match atoms first to ensure same length
        chai_motif_coords, des_motif_coords = match_atoms_by_name(chai_motif, des_motif)
        
        if len(chai_motif_coords) == 0 or len(des_motif_coords) == 0:
            criterion_2_rmsd = np.nan
        elif len(chai_motif_coords) != len(des_motif_coords):
            criterion_2_rmsd = np.nan
        else:
            aligner = _get_aligner(des_motif_coords, chai_motif_coords)
            des_motif_aligned = aligner(des_motif_coords)
            criterion_2_rmsd = _rmsd(chai_motif_coords, des_motif_aligned)
    else:
        criterion_2_rmsd = np.nan
    
    out['criterion_2_metric'] = criterion_2_rmsd
    out['criterion_2'] = criterion_2_rmsd < 1.0
    
    # Criterion 3: CA RMSD (chai vs design, CA-aligned)
    ref_ca = _get_ca_atoms(ref_struct)
    des_ca = _get_ca_atoms(des_struct)
    chai_ca = _get_ca_atoms(chai_struct)
    
    if len(des_ca) > 0 and len(chai_ca) > 0:
        # Match CA atoms by (chain_id, res_id)
        def get_ca_map(ca_atoms):
            ca_map = {}
            for i, (chain_id, res_id) in enumerate(zip(ca_atoms.chain_id, ca_atoms.res_id)):
                if isinstance(res_id, tuple):
                    pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
                else:
                    pdb_res_id = res_id
                key = (str(chain_id), int(pdb_res_id))
                if key not in ca_map:  # Keep first occurrence
                    ca_map[key] = ca_atoms.coord[i]
            return ca_map
        
        des_ca_map = get_ca_map(des_ca)
        chai_ca_map = get_ca_map(chai_ca)
        
        # Find common residues
        common_keys = sorted(set(des_ca_map.keys()) & set(chai_ca_map.keys()))
        if len(common_keys) >= 3:
            des_ca_coords = np.array([des_ca_map[k] for k in common_keys])
            chai_ca_coords = np.array([chai_ca_map[k] for k in common_keys])
            
            aligner = _get_aligner(des_ca_coords, chai_ca_coords)
            des_ca_aligned = aligner(des_ca_coords)
            criterion_3_rmsd = _rmsd(chai_ca_coords, des_ca_aligned)
        else:
            criterion_3_rmsd = np.nan
    else:
        criterion_3_rmsd = np.nan
    
    out['criterion_3_metric'] = criterion_3_rmsd
    out['criterion_3'] = criterion_3_rmsd < 1.5
    
    # Criterion 4: Backbone-ligand clash in design
    ref_ligand = _get_ligand_atoms(ref_struct)
    des_backbone = _get_backbone_atoms(des_struct)
    
    if len(ref_ligand) > 0 and len(des_backbone) > 0:
        # Match motif atoms first to ensure same length for alignment
        ref_motif_coords, des_motif_coords = match_atoms_by_name(ref_motif, des_motif)
        
        if len(ref_motif_coords) == 0 or len(des_motif_coords) == 0 or len(ref_motif_coords) != len(des_motif_coords):
            out['criterion_4_metric'] = np.nan
            out['criterion_4'] = False
        else:
        # Align design motif to reference motif
            aligner = _get_aligner(des_motif_coords, ref_motif_coords)
        ref_ligand_aligned = aligner(ref_ligand.coord)
        
        # Compute distances
        distances = scipy.spatial.distance.cdist(
            ref_ligand_aligned,
            des_backbone.coord
        )
        min_dist = float(distances.min())
        out['criterion_4_metric'] = min_dist
        out['criterion_4'] = min_dist > 2.0
    else:
        out['criterion_4_metric'] = np.nan
        out['criterion_4'] = False
    
    # Criterion 5: Backbone-ligand clash in chai
    des_ligand = _get_ligand_atoms(des_struct)
    chai_backbone = _get_backbone_atoms(chai_struct)
    
    if len(des_ligand) > 0 and len(chai_backbone) > 0:
        # Match motif atoms first to ensure same length for alignment
        chai_motif_coords, des_motif_coords = match_atoms_by_name(chai_motif, des_motif)
        
        if len(chai_motif_coords) == 0 or len(des_motif_coords) == 0 or len(chai_motif_coords) != len(des_motif_coords):
            out['criterion_5_metric'] = np.nan
            out['criterion_5'] = False
        else:
        # Align design motif to chai motif
            aligner = _get_aligner(des_motif_coords, chai_motif_coords)
        des_ligand_aligned = aligner(des_ligand.coord)
        
        # Compute distances
        distances = scipy.spatial.distance.cdist(
            des_ligand_aligned,
            chai_backbone.coord
        )
        min_dist = float(distances.min())
        out['criterion_5_metric'] = min_dist
        out['criterion_5'] = min_dist > 2.0
    else:
        out['criterion_5_metric'] = np.nan
        out['criterion_5'] = False
    
    # Criterion 6: Chai-1 plDDT
    out['criterion_6_metric'] = np.nan if chai_plddt is None else chai_plddt
    out['criterion_6'] = False if chai_plddt is None else (chai_plddt > 0.7)
    
    # RFD2 Success Metrics
    
    # catalytic_heavy_atom_rmsd: Heavy atom RMSD of catalytic residues (chai vs design)
    # aligned to Chai-1 predicted motif backbone N, CA, C
    if len(chai_motif) > 0 and len(des_motif) > 0:
        # Get motif backbone atoms (N, CA, C) for alignment
        chai_motif_bb = _get_motif_atoms(chai_struct, motif_residues, atom_names=["N", "CA", "C"])
        des_motif_bb = _get_motif_atoms(des_struct, motif_residues, atom_names=["N", "CA", "C"])
        
        if len(chai_motif_bb) >= 3 and len(des_motif_bb) >= 3:
            # Match backbone atoms first to ensure same length for alignment
            chai_motif_bb_coords, des_motif_bb_coords = match_atoms_by_name(chai_motif_bb, des_motif_bb)
            
            if len(chai_motif_bb_coords) == 0 or len(des_motif_bb_coords) == 0 or len(chai_motif_bb_coords) != len(des_motif_bb_coords):
                catalytic_heavy_atom_rmsd = np.nan
            else:
            # Align design motif backbone to chai motif backbone (chai is reference)
                aligner = _get_aligner(des_motif_bb_coords, chai_motif_bb_coords)
            
            # Match catalytic residue heavy atoms by (chain_id, res_id, atom_name)
            def match_catalytic_atoms(atoms1, atoms2):
                """Match atoms by (chain_id, res_id, atom_name)."""
                coords1 = []
                coords2 = []
                map1 = {}
                for i, (chain_id, res_id, atom_name) in enumerate(
                    zip(atoms1.chain_id, atoms1.res_id, atoms1.atom_name)
                ):
                    if isinstance(res_id, tuple):
                        pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
                    else:
                        pdb_res_id = res_id
                    key = (str(chain_id), int(pdb_res_id), str(atom_name))
                    if key not in map1:
                        map1[key] = atoms1.coord[i]
                
                for i, (chain_id, res_id, atom_name) in enumerate(
                    zip(atoms2.chain_id, atoms2.res_id, atoms2.atom_name)
                ):
                    if isinstance(res_id, tuple):
                        pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
                    else:
                        pdb_res_id = res_id
                    key = (str(chain_id), int(pdb_res_id), str(atom_name))
                    if key in map1:
                        coords1.append(map1[key])
                        coords2.append(atoms2.coord[i])
                
                return np.array(coords1), np.array(coords2)
            
            chai_coords, des_coords = match_catalytic_atoms(chai_motif, des_motif)
            if len(chai_coords) > 0 and len(des_coords) > 0:
                # Align design coordinates to chai reference frame
                des_coords_aligned = aligner(des_coords)
                catalytic_heavy_atom_rmsd = _rmsd(chai_coords, des_coords_aligned)
            else:
                catalytic_heavy_atom_rmsd = np.nan
        else:
            catalytic_heavy_atom_rmsd = np.nan
    else:
        catalytic_heavy_atom_rmsd = np.nan
    
    out['catalytic_heavy_atom_rmsd'] = catalytic_heavy_atom_rmsd
    
    # ligand_clash_1_5A: Ligand clash detection with 1.5 Å threshold
    # Check ligand-backbone distance in chai structure (similar to criterion_5 but with 1.5 Å threshold)
    des_ligand = _get_ligand_atoms(des_struct)
    chai_backbone = _get_backbone_atoms(chai_struct)
    
    if len(des_ligand) > 0 and len(chai_backbone) > 0:
        # Match motif atoms first to ensure same length for alignment
        chai_motif_coords, des_motif_coords = match_atoms_by_name(chai_motif, des_motif)
        
        if len(chai_motif_coords) == 0 or len(des_motif_coords) == 0 or len(chai_motif_coords) != len(des_motif_coords):
            out['ligand_clash_1_5A_metric'] = np.nan
            out['ligand_clash_1_5A'] = False
        else:
        # Align design motif to chai motif (same as criterion_5)
            aligner = _get_aligner(des_motif_coords, chai_motif_coords)
        des_ligand_aligned = aligner(des_ligand.coord)
        
        # Compute distances
        distances = scipy.spatial.distance.cdist(
            des_ligand_aligned,
            chai_backbone.coord
        )
        min_dist_1_5A = float(distances.min())
        # Count clashes (< 1.5 Å)
        clash_count = int(np.sum(distances < 1.5))
        out['ligand_clash_1_5A_metric'] = min_dist_1_5A
        out['ligand_clash_1_5A'] = min_dist_1_5A > 1.5
        out['ligand_clash_count_1_5A'] = clash_count
    else:
        out['ligand_clash_1_5A_metric'] = np.nan
        out['ligand_clash_1_5A'] = False
        out['ligand_clash_count_1_5A'] = 0
    
    # rfd2_success: Combined success criterion (RFD2 paper standard)
    # - catalytic_heavy_atom_rmsd < 1.5 Å
    # - ligand_clash_count_1_5A == 0 (no clashes < 1.5 Å)
    catalytic_pass = not np.isnan(catalytic_heavy_atom_rmsd) and catalytic_heavy_atom_rmsd < 1.5
    clash_count = out.get('ligand_clash_count_1_5A', 0)
    ligand_pass = clash_count == 0
    out['rfd2_success'] = catalytic_pass and ligand_pass
    
    return out


def compute_default_metrics(
    des_pdb: str | Path,
    motif_residues: List[str],
) -> Dict[str, float]:
    """
    Compute default metrics (clash detection).
    
    Args:
        des_pdb: Design structure
        motif_residues: List of motif residue identifiers
        
    Returns:
        Dictionary with clash metrics
    """
    out = {}
    
    des_struct = _read_structure(des_pdb)
    
    # Get all heavy atoms (excluding backbone-backbone)
    heavy_mask = (des_struct.element != "H")
    heavy_atoms = des_struct[heavy_mask]
    
    # Compute pairwise distances
    coords = heavy_atoms.coord
    distances = scipy.spatial.distance.pdist(coords)
    
    # Minimum distance
    min_dist = float(distances.min()) if len(distances) > 0 else np.nan
    out['res_to_res_min_dist'] = min_dist
    
    # Clash detection (threshold: 2.0 Å, roughly 2 VDW radii)
    clash_threshold = 2.0
    n_clashes = int(np.sum(distances < clash_threshold)) if len(distances) > 0 else 0
    out['n_pair_clash'] = n_clashes
    
    return out


def compute_backbone_metrics(
    des_pdb: str | Path,
    motif_residues: List[str],
) -> Dict[str, float]:
    """
    Compute backbone metrics (ligand distance, secondary structure).
    
    Args:
        des_pdb: Design structure
        motif_residues: List of motif residue identifiers
        
    Returns:
        Dictionary with backbone metrics
    """
    out = {}
    
    des_struct = _read_structure(des_pdb)
    
    # Ligand distance metrics
    ligand_atoms = _get_ligand_atoms(des_struct)
    
    if len(ligand_atoms) > 0:
        # CA distance to ligand
        ca_atoms = _get_ca_atoms(des_struct)
        if len(ca_atoms) > 0:
            distances = scipy.spatial.distance.cdist(
                ca_atoms.coord,
                ligand_atoms.coord
            )
            min_dist_per_res = distances.min(axis=1)
            out['ligand_dist_des_c-alpha'] = min_dist_per_res.tolist()
            out['ligand_dist_des_c-alpha_min'] = float(min_dist_per_res.min())
        
        # Backbone (N,CA,C) distance to ligand
        backbone_atoms = _get_backbone_atoms(des_struct)
        if len(backbone_atoms) > 0:
            distances = scipy.spatial.distance.cdist(
                backbone_atoms.coord,
                ligand_atoms.coord
            )
            min_dist_per_res = distances.min(axis=1)
            out['ligand_dist_des_ncac'] = min_dist_per_res.tolist()
            out['ligand_dist_des_ncac_min'] = float(min_dist_per_res.min())
    
    # Secondary structure (requires mdtraj)
    try:
        import mdtraj as md
        traj = md.load(str(des_pdb))
        ss = md.compute_dssp(traj, simplified=True)
        coil_percent = float(np.mean(ss == 'C'))
        helix_percent = float(np.mean(ss == 'H'))
        strand_percent = float(np.mean(ss == 'E'))
        ss_percent = helix_percent + strand_percent
        
        out['non_coil_percent'] = ss_percent
        out['coil_percent'] = coil_percent
        out['helix_percent'] = helix_percent
        out['strand_percent'] = strand_percent
        
        # Radius of gyration
        rg = md.compute_rg(traj)[0]
        out['radius_of_gyration'] = float(rg)
    except ImportError:
        warnings.warn("mdtraj not available, skipping secondary structure metrics")
    except Exception as e:
        warnings.warn(f"Failed to compute secondary structure metrics: {e}")
    
    return out


def compute_sidechain_metrics(
    ref_pdb: str | Path,
    des_pdb: str | Path,
    chai_pdb: str | Path,
    motif_residues: List[str],
) -> Dict[str, float]:
    """
    Compute sidechain RMSD metrics.
    
    Args:
        ref_pdb: Reference structure
        des_pdb: Design structure
        chai_pdb: Chai-1 refold structure
        motif_residues: List of motif residue identifiers
        
    Returns:
        Dictionary with sidechain RMSD metrics
    """
    out = {}
    
    ref_struct = _read_structure(ref_pdb)
    des_struct = _read_structure(des_pdb)
    chai_struct = _read_structure(chai_pdb)
    
    # Get motif sidechain atoms (all heavy atoms except backbone)
    def get_sidechain_atoms(struct, motif_res):
        motif_atoms = _get_motif_atoms(struct, motif_res)
        # Exclude backbone atoms
        bb_mask = np.isin(motif_atoms.atom_name, ["N", "CA", "C", "O"])
        return motif_atoms[~bb_mask]
    
    ref_sc = get_sidechain_atoms(ref_struct, motif_residues)
    des_sc = get_sidechain_atoms(des_struct, motif_residues)
    chai_sc = get_sidechain_atoms(chai_struct, motif_residues)
    
    # Motif-backbone-aligned sidechain RMSD (design vs ref)
    if len(ref_sc) > 0 and len(des_sc) > 0:
        ref_motif_bb = _get_motif_atoms(ref_struct, motif_residues, atom_names=["N", "CA", "C"])
        des_motif_bb = _get_motif_atoms(des_struct, motif_residues, atom_names=["N", "CA", "C"])
        
        if len(ref_motif_bb) >= 3 and len(des_motif_bb) >= 3:
            # Match backbone atoms first to ensure same length for alignment
            ref_motif_bb_coords, des_motif_bb_coords = match_atoms_by_name(ref_motif_bb, des_motif_bb)
            
            if len(ref_motif_bb_coords) == 0 or len(des_motif_bb_coords) == 0 or len(ref_motif_bb_coords) != len(des_motif_bb_coords):
                out['motif_sidechain_rmsd'] = np.nan
            else:
                aligner = _get_aligner(des_motif_bb_coords, ref_motif_bb_coords)
            des_sc_aligned = aligner(des_sc.coord)
            
            # Match atoms by atom_name
            ref_names = ref_sc.atom_name.tolist()
            des_names = des_sc.atom_name.tolist()
            common_names = sorted(set(ref_names) & set(des_names))
            
            if len(common_names) > 0:
                ref_coords = []
                des_coords = []
                for name in common_names:
                    ref_idx = ref_names.index(name)
                    des_idx = des_names.index(name)
                    ref_coords.append(ref_sc.coord[ref_idx])
                    des_coords.append(des_sc_aligned[des_idx])
                
                if len(ref_coords) >= 1:
                    ref_coords = np.array(ref_coords)
                    des_coords = np.array(des_coords)
                    out['sidechain_rmsd_des_ref'] = _rmsd(ref_coords, des_coords)
    
    # Motif-backbone-aligned sidechain RMSD (chai vs ref)
    if len(ref_sc) > 0 and len(chai_sc) > 0:
        ref_motif_bb = _get_motif_atoms(ref_struct, motif_residues, atom_names=["N", "CA", "C"])
        chai_motif_bb = _get_motif_atoms(chai_struct, motif_residues, atom_names=["N", "CA", "C"])
        
        if len(ref_motif_bb) >= 3 and len(chai_motif_bb) >= 3:
            # Match backbone atoms first to ensure same length for alignment
            ref_motif_bb_coords, chai_motif_bb_coords = match_atoms_by_name(ref_motif_bb, chai_motif_bb)
            
            if len(ref_motif_bb_coords) == 0 or len(chai_motif_bb_coords) == 0 or len(ref_motif_bb_coords) != len(chai_motif_bb_coords):
                # Skip if no matching atoms
                pass
            else:
                aligner = _get_aligner(chai_motif_bb_coords, ref_motif_bb_coords)
            chai_sc_aligned = aligner(chai_sc.coord)
            
            # Match atoms by atom_name
            ref_names = ref_sc.atom_name.tolist()
            chai_names = chai_sc.atom_name.tolist()
            common_names = sorted(set(ref_names) & set(chai_names))
            
            if len(common_names) > 0:
                ref_coords = []
                chai_coords = []
                for name in common_names:
                    ref_idx = ref_names.index(name)
                    chai_idx = chai_names.index(name)
                    ref_coords.append(ref_sc.coord[ref_idx])
                    chai_coords.append(chai_sc_aligned[chai_idx])
                
                if len(ref_coords) >= 1:
                    ref_coords = np.array(ref_coords)
                    chai_coords = np.array(chai_coords)
                    out['sidechain_rmsd_chai_ref'] = _rmsd(ref_coords, chai_coords)
    
    return out
