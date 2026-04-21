"""
Utilities for the ligand-pocket interface benchmark.

The interface benchmark is a constrained variant of ligand-binding protein
design: only protein residues within a ligand-distance cutoff are redesigned.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import biotite.structure.io as io
import numpy as np
from biotite.structure.io import pdb, pdbx


DEFAULT_INTERFACE_POCKET_CUTOFF = 3.5
WATER_RES_NAMES = {"HOH", "WAT"}


def _load_structure(struct_path: Path, extra_fields: Optional[list[str]] = None):
    suffix = struct_path.suffix.lower()
    extra_fields = extra_fields or []
    if suffix == ".pdb":
        return pdb.PDBFile.read(str(struct_path)).get_structure(
            model=1,
            extra_fields=extra_fields,
        )
    if suffix in {".cif", ".mmcif"}:
        return pdbx.get_structure(
            pdbx.CIFFile.read(str(struct_path)),
            model=1,
            extra_fields=extra_fields,
        )
    raise ValueError(f"Unsupported structure format: {struct_path}")


def _residue_key(chain_id, res_id) -> str:
    return f"{str(chain_id)}{int(res_id)}"


def _iter_structure_paths(input_dir: Path) -> List[Path]:
    struct_paths = sorted(input_dir.glob("*.cif")) + sorted(input_dir.glob("*.pdb"))
    return [p for p in struct_paths if p.is_file()]


def calculate_pocket_residues_from_ligand_proximity(
    struct_path: Path | str,
    pocket_cutoff: float = DEFAULT_INTERFACE_POCKET_CUTOFF,
    design_chain: Optional[str] = None,
) -> List[str]:
    """
    Return ordered protein residue IDs (e.g. "A42") within `pocket_cutoff`
    angstrom of any non-water ligand atom.
    """
    struct_path = Path(struct_path)
    atom_array = _load_structure(struct_path)

    protein_atoms = atom_array[~atom_array.hetero]
    if design_chain is not None and str(design_chain).strip():
        design_chain = str(design_chain).strip()
        protein_atoms = protein_atoms[protein_atoms.chain_id == design_chain]

    ligand_mask = atom_array.hetero & (~np.isin(atom_array.res_name, list(WATER_RES_NAMES)))
    ligand_atoms = atom_array[ligand_mask]

    if len(protein_atoms) == 0:
        if design_chain is None:
            raise ValueError(f"No protein atoms found in {struct_path}")
        raise ValueError(f"No protein atoms found on design chain '{design_chain}' in {struct_path}")
    if len(ligand_atoms) == 0:
        raise ValueError(
            f"No non-water ligand atoms found in {struct_path}. "
            "Interface benchmark requires a protein-ligand complex."
        )

    diff = protein_atoms.coord[:, None, :] - ligand_atoms.coord[None, :, :]
    atom_is_close = np.any(np.sum(diff * diff, axis=-1) <= float(pocket_cutoff) ** 2, axis=1)

    pocket_residues: List[str] = []
    seen = set()
    for idx in np.flatnonzero(atom_is_close):
        key = _residue_key(protein_atoms.chain_id[idx], protein_atoms.res_id[idx])
        if key in seen:
            continue
        seen.add(key)
        pocket_residues.append(key)

    if not pocket_residues:
        raise ValueError(
            f"No protein residues were found within {pocket_cutoff:.2f}A of the ligand in {struct_path}"
        )

    return pocket_residues


def calculate_fixed_residues_from_ligand_proximity(
    struct_path: Path | str,
    pocket_cutoff: float = DEFAULT_INTERFACE_POCKET_CUTOFF,
    design_chain: Optional[str] = None,
) -> List[str]:
    """
    Return ordered protein residue IDs to keep fixed for LigandMPNN.
    """
    struct_path = Path(struct_path)
    atom_array = _load_structure(struct_path)
    protein_atoms = atom_array[~atom_array.hetero]
    redesign_residues = set(
        calculate_pocket_residues_from_ligand_proximity(
            struct_path,
            pocket_cutoff=pocket_cutoff,
            design_chain=design_chain,
        )
    )

    fixed_residues: List[str] = []
    seen = set()
    for chain_id, res_id in zip(protein_atoms.chain_id, protein_atoms.res_id):
        key = _residue_key(chain_id, res_id)
        if key in seen or key in redesign_residues:
            continue
        seen.add(key)
        fixed_residues.append(key)
    return fixed_residues


def annotate_interface_pocket_b_factors(
    struct_path: Path | str,
    pocket_cutoff: float = DEFAULT_INTERFACE_POCKET_CUTOFF,
    redesign_b_factor: float = 0.0,
    fixed_b_factor: float = 1.0,
    design_chain: Optional[str] = None,
) -> Dict[str, object]:
    """
    Overwrite B-factors so LigandMPNN redesigns only pocket residues.
    """
    struct_path = Path(struct_path)
    atom_array = _load_structure(struct_path, extra_fields=["b_factor", "occupancy"])
    pocket_residues = set(
        calculate_pocket_residues_from_ligand_proximity(
            struct_path,
            pocket_cutoff=pocket_cutoff,
            design_chain=design_chain,
        )
    )

    if not hasattr(atom_array, "b_factor"):
        atom_array.b_factor = np.full(len(atom_array), fixed_b_factor, dtype=np.float32)
    else:
        atom_array.b_factor = np.asarray(atom_array.b_factor, dtype=np.float32)
        atom_array.b_factor[:] = fixed_b_factor

    protein_mask = ~atom_array.hetero
    protein_keys = np.char.add(
        atom_array.chain_id.astype(str),
        np.asarray(atom_array.res_id, dtype=str),
    )
    redesign_mask = protein_mask & np.isin(protein_keys, list(pocket_residues))
    atom_array.b_factor[redesign_mask] = redesign_b_factor

    io.save_structure(str(struct_path), atom_array)
    return {
        "design_name": struct_path.stem,
        "pocket_cutoff": float(pocket_cutoff),
        "pocket_residue_count": len(pocket_residues),
        "pocket_residues": sorted(pocket_residues),
    }


def build_interface_info_csv(
    input_dir: Path | str,
    output_csv: Path | str,
    pocket_cutoff: float = DEFAULT_INTERFACE_POCKET_CUTOFF,
    design_chain_map: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Persist pocket residue definitions used by the interface benchmark.
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for struct_path in _iter_structure_paths(input_dir):
        design_chain = None
        if design_chain_map is not None:
            design_chain = design_chain_map.get(struct_path.stem, None)
        pocket_residues = calculate_pocket_residues_from_ligand_proximity(
            struct_path,
            pocket_cutoff=pocket_cutoff,
            design_chain=design_chain,
        )
        rows.append(
            {
                "design_name": struct_path.stem,
                "design_chain": design_chain or "",
                "pocket_cutoff": f"{float(pocket_cutoff):.3f}",
                "pocket_residue_count": str(len(pocket_residues)),
                "pocket_residues": ";".join(pocket_residues),
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "design_name",
                "design_chain",
                "pocket_cutoff",
                "pocket_residue_count",
                "pocket_residues",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def load_interface_info_csv(csv_path: Path | str) -> Dict[str, Dict[str, object]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Interface info CSV not found: {csv_path}")

    info: Dict[str, Dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            design_name = Path(str(row["design_name"]).strip()).stem
            pocket_residues = [
                residue.strip()
                for residue in str(row.get("pocket_residues", "")).split(";")
                if residue.strip()
            ]
            info[design_name] = {
                "design_name": design_name,
                "design_chain": str(row.get("design_chain", "")).strip() or None,
                "pocket_cutoff": float(row.get("pocket_cutoff", DEFAULT_INTERFACE_POCKET_CUTOFF)),
                "pocket_residue_count": int(row.get("pocket_residue_count", len(pocket_residues))),
                "pocket_residues": pocket_residues,
            }
    return info


def match_name_to_interface_info(
    design_name: str,
    interface_info: Dict[str, Dict[str, object]],
) -> Optional[Dict[str, object]]:
    """
    Match sequence-expanded names like `sample-3` back to `sample`.
    """
    normalized = Path(str(design_name).strip()).stem
    candidates = [normalized]

    base_name = re.sub(r"-\d+$", "", normalized)
    if base_name != normalized:
        candidates.append(base_name)

    for candidate in candidates:
        if candidate in interface_info:
            return interface_info[candidate]
    return None
