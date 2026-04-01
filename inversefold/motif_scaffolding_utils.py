import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from biotite.structure.io import pdb


# Pre-defined chain ranges for each motif problem
# Format: {problem_id: {chain_id: (start_res, end_res)}}
MOTIF_CHAIN_RANGES: Dict[str, Dict[str, tuple[int, int]]] = {
    "01_1LDB": {"A": (1, 21)},
    "02_1ITU": {"A": (1, 24)},
    "03_2CGA": {"A": (1, 11)},
    "04_5WN9": {"A": (1, 20)},
    "05_5ZE9": {"A": (1, 15)},
    "06_6E6R": {"A": (1, 13)},
    "07_6E6R": {"A": (1, 13)},
    "08_7AD5": {"A": (1, 15)},
    "09_7CG5": {"A": (1, 15)},
    "10_7WRK": {"A": (1, 15)},
    "11_3TQB": {"A": (1, 15), "B": (1, 15)},
    "12_4JHW": {"A": (1, 7), "B": (1, 17)},
    "13_4JHW": {"A": (1, 7), "B": (1, 17)},
    "14_5IUS": {"A": (1, 20), "B": (1, 22)},
    "15_7A8S": {"A": (1, 15), "B": (1, 15)},
    "16_7BNY": {"A": (1, 15), "B": (1, 15)},
    "17_7DGW": {"A": (1, 15), "B": (1, 15)},
    "18_7MQQ": {"A": (1, 15), "B": (1, 15)},
    "19_7MQQ": {"A": (1, 15), "B": (1, 15)},
    "20_7UWL": {"A": (1, 11), "B": (1, 11)},
    "21_1B73": {"A": (1, 2), "B": (1, 1), "C": (1, 3)},
    "22_1BCF": {"A": (1, 8), "B": (1, 8), "C": (1, 8), "D": (1, 8)},
    "23_1MPY": {"A": (1, 1), "B": (1, 1), "C": (1, 1), "D": (1, 1), "E": (1, 1), "F": (1, 1)},
    "24_1QY3": {"A": (1, 14), "B": (1, 1), "C": (1, 1)},
    "25_2RKX": {"A": (1, 3), "B": (1, 3), "C": (1, 1), "D": (1, 1), "E": (1, 1), "F": (1, 1), "G": (1, 1), "H": (1, 3)},
    "26_3B5V": {"A": (1, 3), "B": (1, 1), "C": (1, 1), "D": (1, 1), "E": (1, 1), "F": (1, 5), "G": (1, 2), "H": (1, 3)},
    "27_4XOJ": {"A": (1, 1), "B": (1, 1), "C": (1, 3)},
    "28_5YUI": {"A": (1, 5), "B": (1, 3), "C": (1, 3)},
    "29_6CPA": {"A": (1, 4), "B": (1, 1), "C": (1, 1), "D": (1, 1), "E": (1, 1)},
    "30_7UWL": {"A": (1, 11), "B": (1, 11), "C": (1, 11), "D": (1, 10)},
}


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


def _get_motif_chain_range(problem_id: str, chain_id: str) -> tuple[int, int]:
    """
    Get the motif range for a specific chain from the pre-defined MOTIF_CHAIN_RANGES.
    Returns (start, end) in 1-based inclusive numbering.
    """
    if problem_id in MOTIF_CHAIN_RANGES:
        chain_ranges = MOTIF_CHAIN_RANGES[problem_id]
        if chain_id in chain_ranges:
            return chain_ranges[chain_id]
    
    # If not found in pre-defined, raise error
    raise ValueError(
        f"Chain '{chain_id}' not found in pre-defined ranges for problem '{problem_id}'. "
        f"Available chains: {list(MOTIF_CHAIN_RANGES.get(problem_id, {}).keys())}"
    )


def _parse_motif_token_range(token: str) -> Optional[tuple[int, int]]:
    """
    Parse explicit motif range from tokens like 'A1-15' or 'B7'.
    Returns (start, end) in 1-based inclusive numbering, or None if absent.
    """
    token = token.strip()
    match = re.fullmatch(r"[A-Za-z](\d+)(?:-(\d+))?", token)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    if end < start:
        raise ValueError(f"Invalid motif range token '{token}': end < start")
    return (start, end)


def _extract_problem_id_from_pdb_path(pdb_path: Path) -> Optional[str]:
    """
    Extract motif problem ID from PDB path.
    
    Checks multiple sources in order:
    1. Parent directory name (e.g., "/path/to/25_2RKX/25_2RKX_0.pdb" -> "25_2RKX")
    2. Filename prefix (e.g., "25_2RKX_0.pdb" -> "25_2RKX")
    """
    # First, try parent directory name
    parent_name = pdb_path.parent.name
    if parent_name in MOTIF_CHAIN_RANGES:
        return parent_name
    
    # Then, try to extract from filename
    stem = pdb_path.stem
    # Handle patterns like "25_2RKX_0" or "25_2RKX_0-1"
    if "_" in stem:
        parts = stem.split("_")
        # Try to find the longest match that exists in MOTIF_CHAIN_RANGES
        for i in range(len(parts) - 1, 0, -1):
            potential_id = "_".join(parts[:i])
            if potential_id in MOTIF_CHAIN_RANGES:
                return potential_id
    return None


def calculate_fixed_residues_from_motif_placements(
    pdb_path: Path,
    scaffold_row: Any,
    motif_problem_id: Optional[str] = None
) -> List[str]:
    """
    Return motif residues as fixed positions for Ligand/ProteinMPNN input.

    The motif_placements format is: scaffold_length/motif_chain/scaffold_length/motif_chain/...
    Where motif_chain is a single letter (A, B, C, ...) identifying a chain in the reference PDB.
    The actual range of each motif chain is looked up from MOTIF_CHAIN_RANGES.

    Example: "0/A/30/B/45/C/21" means:
    - 0 scaffold residues
    - A chain motif (looked up from PDB, e.g., A1-3)
    - 30 scaffold residues
    - B chain motif (looked up from PDB, e.g., B1-3)
    - 45 scaffold residues
    - C chain motif (looked up from PDB, e.g., C1-1)
    - 21 scaffold residues

    Args:
        pdb_path: Path to the generated scaffold PDB file
        scaffold_row: Row from scaffold_info.csv containing motif_placements
        motif_problem_id: Optional problem ID (e.g., "25_2RKX") to look up chain ranges.
                         If not provided, attempts to extract from pdb_path filename.
    """
    motif_placements = str(scaffold_row.get("motif_placements", "")).strip()
    if not motif_placements:
        raise ValueError("motif_placements is empty")

    # Extract problem_id from scaffold_row if not provided
    if motif_problem_id is None:
        motif_problem_id = str(scaffold_row.get("problem", ""))
        if not motif_problem_id:
            # Try to extract from pdb_path filename
            motif_problem_id = _extract_problem_id_from_pdb_path(pdb_path)

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

    # Parse tokens left-to-right with a running cursor over the designed chain.
    # motif_placements format: scaffold/motif/scaffold/motif/.../scaffold
    # The motif token can be "A" (chain only) or "A1-15" (explicit range).
    cursor = 0
    motif_positions_1based: List[int] = []
    unresolved_motifs: List[tuple[int, str]] = []  # (token_index, chain_id)
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token[0].isdigit():
            # Scaffold length
            scaffold_length = _parse_scaffold_length(token)
            cursor += scaffold_length
            i += 1
        else:
            # Motif chain identifier (e.g., "A", "B", "C")
            chain_id = token[0]  # Single letter chain ID
            
            # Look up the actual motif range from pre-defined ranges
            try:
                explicit_range = _parse_motif_token_range(token)
                if explicit_range is not None:
                    motif_start, motif_end = explicit_range
                else:
                    motif_start, motif_end = _get_motif_chain_range(motif_problem_id, chain_id)
                motif_length = motif_end - motif_start + 1
                
                # Place motif by cumulative designed length, not by reference residue id.
                motif_seq_start = cursor + 1
                motif_positions_1based.extend(range(motif_seq_start, motif_seq_start + motif_length))
                cursor += motif_length
                
                i += 1
            except ValueError:
                # Chain not found, mark as unresolved
                unresolved_motifs.append((i, chain_id))
                cursor += 1  # Add placeholder length
                i += 1

    # Handle unresolved motifs if possible
    if unresolved_motifs:
        if len(unresolved_motifs) == 1:
            idx, chain_id = unresolved_motifs[0]
            # Try to infer from remaining length
            if motif_positions_1based:
                last_motif_end = max(motif_positions_1based)
            else:
                last_motif_end = 0
            inferred_length = total_residues - last_motif_end
            if inferred_length > 0:
                motif_seq_start = last_motif_end + 1
                motif_positions_1based.extend(range(motif_seq_start, motif_seq_start + inferred_length))
            else:
                raise ValueError(
                    f"Cannot infer motif length for unresolved chain '{chain_id}'. "
                    f"motif_placements='{motif_placements}', total_residues={total_residues}"
                )
        else:
            raise ValueError(
                f"Cannot infer motif lengths from motif_placements. "
                f"motif_placements='{motif_placements}', total_residues={total_residues}, "
                f"unresolved_motifs={unresolved_motifs}"
            )

    if motif_positions_1based and max(motif_positions_1based) > total_residues:
        raise ValueError(
            "Motif positions exceed designed chain length. "
            f"max_motif_pos={max(motif_positions_1based)}, total_residues={total_residues}, "
            f"motif_placements='{motif_placements}'"
        )

    # Convert 1-based positions to actual residue identifiers
    fixed_residues = []
    for pos in motif_positions_1based:
        if pos <= len(residue_ids):
            fixed_residues.append(f"{design_chain}{residue_ids[pos - 1]}")
    
    return fixed_residues


def read_motif_chain_ranges_from_pdb(motif_pdb_path: Path) -> Dict[str, tuple[int, int]]:
    """
    Read chain ranges from a motif PDB file.
    Returns a dict mapping chain_id to (start_res, end_res) tuple.
    """
    chain_ranges: Dict[str, tuple[int, int]] = {}
    
    with open(motif_pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21].strip()
                if not chain_id:
                    continue
                resseq = int(line[22:26].strip())
                
                if chain_id not in chain_ranges:
                    chain_ranges[chain_id] = (resseq, resseq)
                else:
                    start, end = chain_ranges[chain_id]
                    chain_ranges[chain_id] = (min(start, resseq), max(end, resseq))
    
    return chain_ranges
