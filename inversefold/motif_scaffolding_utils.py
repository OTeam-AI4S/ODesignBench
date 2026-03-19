import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from biotite.structure.io import pdb


def load_scaffold_info_csv(scaffold_info_csv: str) -> List[Dict[str, str]]:
    """Load and validate motif scaffolding metadata."""
    with open(scaffold_info_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    required_columns = {"sample_num", "motif_placements"}
    missing = required_columns - columns
    if missing:
        raise ValueError(
            f"scaffold_info.csv is missing required columns: {sorted(missing)}"
        )
    return rows


def _extract_sample_num_from_filename(struct_path: Path) -> Optional[int]:
    """
    Extract sample number from filenames like:
    - 01_1LDB_0.pdb
    - 01_1LDB_0-anything.pdb
    """
    stem = struct_path.stem
    # Remove trailing sequence suffix if present: foo_0-7 -> foo_0
    if "-" in stem:
        stem = stem.rsplit("-", 1)[0]
    match = re.search(r"_([0-9]+)$", stem)
    if match:
        return int(match.group(1))
    return None


def match_pdb_to_scaffold_info(struct_path: Path, scaffold_info_df: Any) -> Optional[Any]:
    """Match a design PDB path to one scaffold_info row by sample_num."""
    sample_num = _extract_sample_num_from_filename(struct_path)
    if sample_num is None:
        return None
    for row in scaffold_info_df:
        try:
            if int(str(row.get("sample_num", "")).strip()) == int(sample_num):
                return row
        except ValueError:
            continue
    return None


def _parse_scaffold_length(token: str) -> int:
    """
    Parse scaffold length token.
    Accepts exact lengths ("34") and sampled-range notation ("30-40").
    For ranges, use lower bound as a deterministic fallback.
    """
    t = token.strip()
    if "-" in t:
        left, _right = t.split("-", 1)
        return int(left)
    return int(t)


def _parse_motif_token_length(token: str) -> Optional[int]:
    """
    Parse motif token length from forms:
    - A1-21  -> 21
    - B7     -> 1
    - A      -> None (length unresolved, inferred later if possible)
    """
    t = token.strip()
    if not t or not t[0].isalpha():
        return None
    if len(t) == 1:
        return None
    span = t[1:]
    if "-" in span:
        start_text, end_text = span.split("-", 1)
        return int(end_text) - int(start_text) + 1
    return 1


def calculate_fixed_residues_from_motif_placements(
    pdb_path: Path,
    scaffold_row: Any
) -> List[str]:
    """
    Return motif residues as fixed positions for Ligand/ProteinMPNN input.

    The motif region is reconstructed from scaffold_info `motif_placements` and
    the designed protein chain residue ordering in `pdb_path`.
    """
    motif_placements = str(scaffold_row.get("motif_placements", "")).strip()
    if not motif_placements:
        raise ValueError("motif_placements is empty")

    atom_array = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1, extra_fields=["b_factor"])
    protein_atoms = atom_array[~atom_array.hetero]
    if len(protein_atoms) == 0:
        raise ValueError(f"No protein atoms found in {pdb_path}")

    design_chain = str(protein_atoms.chain_id[0])
    chain_atoms = protein_atoms[protein_atoms.chain_id == design_chain]
    if len(chain_atoms) == 0:
        raise ValueError(f"No atoms found for design chain '{design_chain}' in {pdb_path}")

    residue_ids: List[int] = []
    for res_id in chain_atoms.res_id:
        rid = int(res_id)
        if not residue_ids or residue_ids[-1] != rid:
            residue_ids.append(rid)

    total_residues = len(residue_ids)
    if total_residues == 0:
        raise ValueError(f"No residues found for chain '{design_chain}' in {pdb_path}")

    tokens = [t.strip() for t in motif_placements.strip("/").split("/") if t.strip()]
    if not tokens:
        raise ValueError(f"Invalid motif_placements: '{motif_placements}'")

    scaffold_total = 0
    motif_known_total = 0
    unresolved_motif_indices: List[int] = []
    token_lengths: List[Optional[int]] = []
    token_types: List[str] = []

    for idx, token in enumerate(tokens):
        if token[0].isdigit():
            length = _parse_scaffold_length(token)
            scaffold_total += length
            token_types.append("scaffold")
            token_lengths.append(length)
        else:
            length = _parse_motif_token_length(token)
            token_types.append("motif")
            token_lengths.append(length)
            if length is None:
                unresolved_motif_indices.append(idx)
            else:
                motif_known_total += length

    if unresolved_motif_indices:
        remaining = total_residues - scaffold_total - motif_known_total
        if len(unresolved_motif_indices) == 1 and remaining > 0:
            token_lengths[unresolved_motif_indices[0]] = remaining
        else:
            raise ValueError(
                "Cannot infer motif length from motif_placements. "
                f"motif_placements='{motif_placements}', total_residues={total_residues}, "
                f"scaffold_total={scaffold_total}, known_motif_total={motif_known_total}, "
                f"unresolved_segments={len(unresolved_motif_indices)}"
            )

    # Build 1-based sequence positions for motif segments.
    seq_pos = 1
    motif_positions_1based: List[int] = []
    for seg_type, seg_len in zip(token_types, token_lengths):
        if seg_len is None or seg_len <= 0:
            raise ValueError(
                f"Invalid segment length {seg_len} in motif_placements='{motif_placements}'"
            )
        if seg_type == "motif":
            motif_positions_1based.extend(range(seq_pos, seq_pos + seg_len))
        seq_pos += seg_len

    if motif_positions_1based and max(motif_positions_1based) > total_residues:
        raise ValueError(
            "Motif positions exceed designed chain length. "
            f"max_motif_pos={max(motif_positions_1based)}, total_residues={total_residues}, "
            f"motif_placements='{motif_placements}'"
        )

    fixed_residues = [f"{design_chain}{residue_ids[pos - 1]}" for pos in motif_positions_1based]
    return fixed_residues
