"""
Pocket-aligned ligand RMSD metric.

We define the pocket from a reference (input) structure as all protein residues
with any atom within `pocket_cutoff` Å of the ligand heavy atoms. We then:
1) Find matching CA atoms for those residues in the predicted structure
   by (chain_id, res_id).
2) Compute a Kabsch alignment from predicted pocket CA -> reference pocket CA.
3) Apply the transform to predicted ligand heavy atoms.
4) Compute ligand RMSD vs reference ligand heavy atoms by matching atom_name.

This mirrors the idea of "pocket-aligned ligand" metrics used in enzyme design
benchmarks, while staying robust to global rigid-body differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

import biotite.structure as struc
from biotite.structure.io import pdb, pdbx


_WATER_NAMES = {"HOH", "WAT", "H2O"}


def _read_structure(path: str | Path) -> struc.AtomArray:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".cif":
        cif = pdbx.CIFFile.read(str(path))
        return pdbx.get_structure(cif, model=1)
    if suf == ".pdb":
        return pdb.PDBFile.read(str(path)).get_structure(model=1)
    raise ValueError(f"Unsupported structure format: {path}")


def _is_protein_atom_array(arr: struc.AtomArray) -> np.ndarray:
    """Return mask selecting protein atoms (biotite amino acid filter)."""
    try:
        return struc.filter_amino_acids(arr)
    except Exception:
        # Conservative fallback: treat standard AA residue names as protein
        aa3 = set(struc.info.standard_amino_acids())
        return np.isin(arr.res_name, list(aa3))


def _pick_ligand_residue(arr: struc.AtomArray) -> Optional[Tuple[str, int, str]]:
    """
    Pick a ligand residue identifier (chain_id, res_id, res_name) from a structure.
    Strategy: choose the non-protein, non-water residue with the most heavy atoms.
    Note: res_name is kept for compatibility but matching will be done by chain_id and res_id only.
    """
    is_prot = _is_protein_atom_array(arr)
    is_water = np.isin(arr.res_name, list(_WATER_NAMES))
    is_het = (~is_prot) & (~is_water)
    if not np.any(is_het):
        return None

    het = arr[is_het]
    heavy = het.element != "H"
    het = het[heavy]
    if len(het) == 0:
        return None

    # group by (chain_id, res_id) only - ignore res_name for matching
    keys = list(zip(het.chain_id.tolist(), het.res_id.tolist()))
    counts: Dict[Tuple[str, int], int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    # pick largest group
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    # Get res_name from first atom with this chain_id and res_id (for compatibility)
    best_atoms = het[(het.chain_id == best_key[0]) & (het.res_id == best_key[1])]
    res_name = best_atoms.res_name[0] if len(best_atoms) > 0 else "UNK"
    return (best_key[0], best_key[1], res_name)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal rotation R and translation t such that:
      (P @ R) + t ≈ Q
    where P,Q are (N,3).
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    # Ensure right-handed rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = Qc - Pc @ R
    return R, t


@dataclass
class PocketAlignedLigandResult:
    ligand_rmsd: float
    n_pocket_res: int
    n_ligand_atoms_matched: int
    ligand_resname_ref: str | None
    ligand_resname_pred: str | None


def compute_pocket_aligned_ligand_rmsd(
    reference_structure: str | Path,
    predicted_structure: str | Path,
    pocket_cutoff: float = 6.0,
) -> PocketAlignedLigandResult:
    """
    Compute pocket-aligned ligand RMSD between reference and predicted structure.
    Returns NaNs if ligand/pocket cannot be determined robustly.
    """
    ref = _read_structure(reference_structure)
    pred = _read_structure(predicted_structure)

    ref_lig_id = _pick_ligand_residue(ref)
    pred_lig_id = _pick_ligand_residue(pred)

    if ref_lig_id is None or pred_lig_id is None:
        return PocketAlignedLigandResult(
            ligand_rmsd=float("nan"),
            n_pocket_res=0,
            n_ligand_atoms_matched=0,
            ligand_resname_ref=ref_lig_id[2] if ref_lig_id else None,
            ligand_resname_pred=pred_lig_id[2] if pred_lig_id else None,
        )

    # Match ligands by chain_id and res_id only, ignoring res_name
    # This allows matching when reference has BCA and prediction has LIG2
    ref_lig = ref[
        (ref.chain_id == ref_lig_id[0])
        & (ref.res_id == ref_lig_id[1])
        & (ref.element != "H")
        & (~_is_protein_atom_array(ref))
        & (~np.isin(ref.res_name, list(_WATER_NAMES)))
    ]
    pred_lig = pred[
        (pred.chain_id == pred_lig_id[0])
        & (pred.res_id == pred_lig_id[1])
        & (pred.element != "H")
        & (~_is_protein_atom_array(pred))
        & (~np.isin(pred.res_name, list(_WATER_NAMES)))
    ]
    if len(ref_lig) == 0 or len(pred_lig) == 0:
        return PocketAlignedLigandResult(
            ligand_rmsd=float("nan"),
            n_pocket_res=0,
            n_ligand_atoms_matched=0,
            ligand_resname_ref=ref_lig_id[2],
            ligand_resname_pred=pred_lig_id[2],
        )

    # Define pocket residues in reference (protein atoms within cutoff of ligand)
    ref_prot = ref[_is_protein_atom_array(ref)]
    # distance from each protein atom to any ligand atom
    # (M,3) vs (L,3) -> (M,L)
    d = np.linalg.norm(ref_prot.coord[:, None, :] - ref_lig.coord[None, :, :], axis=-1)
    close = (d.min(axis=1) <= pocket_cutoff)
    if not np.any(close):
        return PocketAlignedLigandResult(
            ligand_rmsd=float("nan"),
            n_pocket_res=0,
            n_ligand_atoms_matched=0,
            ligand_resname_ref=ref_lig_id[2],
            ligand_resname_pred=pred_lig_id[2],
        )

    pocket_atoms = ref_prot[close]
    # pocket residues: unique (chain_id, res_id)
    pocket_res_keys = sorted(set(zip(pocket_atoms.chain_id.tolist(), pocket_atoms.res_id.tolist())))
    if len(pocket_res_keys) < 3:
        # Kabsch alignment is unstable with too few points
        return PocketAlignedLigandResult(
            ligand_rmsd=float("nan"),
            n_pocket_res=len(pocket_res_keys),
            n_ligand_atoms_matched=0,
            ligand_resname_ref=ref_lig_id[2],
            ligand_resname_pred=pred_lig_id[2],
        )

    # Build CA coordinate pairs for pocket residues
    ref_ca = []
    pred_ca = []
    for chain_id, res_id in pocket_res_keys:
        ref_sel = (
            (ref.chain_id == chain_id)
            & (ref.res_id == res_id)
            & (ref.atom_name == "CA")
        )
        pred_sel = (
            (pred.chain_id == chain_id)
            & (pred.res_id == res_id)
            & (pred.atom_name == "CA")
        )
        if np.any(ref_sel) and np.any(pred_sel):
            ref_ca.append(ref.coord[np.where(ref_sel)[0][0]])
            pred_ca.append(pred.coord[np.where(pred_sel)[0][0]])

    if len(ref_ca) < 3 or len(pred_ca) < 3:
        return PocketAlignedLigandResult(
            ligand_rmsd=float("nan"),
            n_pocket_res=len(pocket_res_keys),
            n_ligand_atoms_matched=0,
            ligand_resname_ref=ref_lig_id[2],
            ligand_resname_pred=pred_lig_id[2],
        )

    ref_ca = np.asarray(ref_ca, dtype=float)
    pred_ca = np.asarray(pred_ca, dtype=float)
    R, t = _kabsch(pred_ca, ref_ca)

    # Match ligand atoms by atom_name intersection
    ref_names = ref_lig.atom_name.tolist()
    pred_names = pred_lig.atom_name.tolist()
    common = sorted(set(ref_names).intersection(pred_names))
    if not common:
        return PocketAlignedLigandResult(
            ligand_rmsd=float("nan"),
            n_pocket_res=len(pocket_res_keys),
            n_ligand_atoms_matched=0,
            ligand_resname_ref=ref_lig_id[2],
            ligand_resname_pred=pred_lig_id[2],
        )

    ref_coords = []
    pred_coords = []
    for name in common:
        # if duplicates exist, pick first occurrence
        ref_i = ref_names.index(name)
        pred_i = pred_names.index(name)
        ref_coords.append(ref_lig.coord[ref_i])
        pred_coords.append(pred_lig.coord[pred_i])

    ref_coords = np.asarray(ref_coords, dtype=float)
    pred_coords = np.asarray(pred_coords, dtype=float)
    pred_coords_aligned = pred_coords @ R + t
    rmsd = float(np.sqrt(np.mean(np.sum((pred_coords_aligned - ref_coords) ** 2, axis=1))))

    return PocketAlignedLigandResult(
        ligand_rmsd=rmsd,
        n_pocket_res=len(pocket_res_keys),
        n_ligand_atoms_matched=len(common),
        ligand_resname_ref=ref_lig_id[2],
        ligand_resname_pred=pred_lig_id[2],
    )

