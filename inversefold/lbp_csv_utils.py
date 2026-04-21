"""
Utilities for processing LBP/interface chain-role CSV input.

Required CSV columns:
- design_name: Structure identifier (filename stem or filename with extension)
- target_chain: Chain ID of the ligand/conditioning chain
- design_chain: Chain ID of the redesigned protein chain
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import re

import pandas as pd
from biotite.structure.io import pdb, pdbx


REQUIRED_COLUMNS = {"design_name", "target_chain", "design_chain"}


def load_lbp_info_csv(csv_path: str) -> pd.DataFrame:
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"LBP info CSV not found: {csv_path}")

    df = pd.read_csv(csv_path_obj)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"LBP info CSV missing required columns: {missing_str}")

    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    empty_rows = df[
        (df["design_name"] == "")
        | (df["target_chain"] == "")
        | (df["design_chain"] == "")
    ]
    if len(empty_rows) > 0:
        raise ValueError(
            "LBP info CSV contains empty required values in "
            f"{len(empty_rows)} row(s)."
        )

    normalized = df["design_name"].map(lambda x: Path(x).stem)
    if normalized.duplicated().any():
        dup = sorted(set(normalized[normalized.duplicated()].tolist()))
        raise ValueError(
            "LBP info CSV has duplicate design_name entries (after stem normalization): "
            + ", ".join(dup[:10])
        )

    return df


def match_pdb_to_lbp_info(struct_path: Path, lbp_df: pd.DataFrame) -> Optional[Dict[str, str]]:
    struct_name = struct_path.name
    struct_stem = struct_path.stem

    candidate_names = [struct_name, struct_stem]
    normalized = lbp_df["design_name"].map(lambda x: Path(x).stem)

    base_stem = re.sub(r"-\d+$", "", struct_stem)
    if base_stem != struct_stem:
        candidate_names.append(base_stem)

    exact = lbp_df.iloc[0:0]
    for candidate in candidate_names:
        exact = lbp_df[lbp_df["design_name"] == candidate]
        if len(exact) > 0:
            break
        exact = lbp_df[normalized == Path(candidate).stem]
        if len(exact) > 0:
            break

    if len(exact) == 0:
        return None

    row = exact.iloc[0]
    return {
        "design_name": str(row["design_name"]).strip(),
        "target_chain": str(row["target_chain"]).strip(),
        "design_chain": str(row["design_chain"]).strip(),
    }


def _residue_key(chain_id: str, res_id) -> str:
    if isinstance(res_id, tuple):
        numeric = res_id[1] if len(res_id) > 1 else res_id[0]
    else:
        numeric = res_id
    return f"{chain_id}{int(numeric)}"


def calculate_fixed_residues_from_chain_roles(
    pdb_path: Path,
    design_chain: str,
    target_chain: str,
) -> List[str]:
    """
    Return fixed residues for LBP/interface:
    - redesign only residues on design_chain
    - fix all protein residues on every other protein chain
    - target_chain may be a ligand chain and is validated against all chains
    """
    if pdb_path.suffix.lower() == ".pdb":
        atom_array = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1)
    elif pdb_path.suffix.lower() in [".cif", ".mmcif"]:
        cif_file = pdbx.CIFFile.read(str(pdb_path))
        atom_array = pdbx.get_structure(cif_file, model=1)
    else:
        raise ValueError(f"Unsupported structure format for LBP chain-role parsing: {pdb_path.suffix}")

    all_chain_ids = set(map(str, atom_array.chain_id.tolist()))
    protein_atoms = atom_array[~atom_array.hetero]
    protein_chain_ids = set(map(str, protein_atoms.chain_id.tolist()))

    if target_chain not in all_chain_ids:
        raise ValueError(
            f"Target chain '{target_chain}' not found in {pdb_path.name}. "
            f"Available chains: {sorted(all_chain_ids)}"
        )
    if design_chain not in protein_chain_ids:
        raise ValueError(
            f"Design chain '{design_chain}' not found among protein chains in {pdb_path.name}. "
            f"Available protein chains: {sorted(protein_chain_ids)}"
        )

    fixed_residues: List[str] = []
    seen = set()
    for chain_id, res_id in zip(protein_atoms.chain_id, protein_atoms.res_id):
        chain = str(chain_id)
        if chain == design_chain:
            continue
        key = _residue_key(chain, res_id)
        if key in seen:
            continue
        seen.add(key)
        fixed_residues.append(key)

    return fixed_residues
