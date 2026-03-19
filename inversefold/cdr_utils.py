"""
CDR Utilities for Antibody Inverse Folding

Provides functions to calculate fixed residues (scaffold) based on CDR information.
For antibody design, we fix all residues EXCEPT the 3 CDR loops.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser
import warnings
warnings.filterwarnings('ignore')


def parse_cdr_ranges(cdr_info_row: pd.Series) -> Dict[str, Optional[Tuple[int, int]]]:
    """
    Parse CDR ranges from a row in CDR info CSV.
    
    Args:
        cdr_info_row: Pandas Series with CDR information
        
    Returns:
        Dictionary with keys: 'h_cdr1', 'h_cdr2', 'h_cdr3', 'l_cdr1', 'l_cdr2', 'l_cdr3'
        Values are (start, end) tuples (inclusive) or None if not present
    """
    cdr_ranges = {}
    
    # Heavy chain CDRs
    if pd.notna(cdr_info_row.get('h_cdr1_start')) and pd.notna(cdr_info_row.get('h_cdr1_end')):
        cdr_ranges['h_cdr1'] = (int(cdr_info_row['h_cdr1_start']), int(cdr_info_row['h_cdr1_end']))
    else:
        cdr_ranges['h_cdr1'] = None
    
    if pd.notna(cdr_info_row.get('h_cdr2_start')) and pd.notna(cdr_info_row.get('h_cdr2_end')):
        cdr_ranges['h_cdr2'] = (int(cdr_info_row['h_cdr2_start']), int(cdr_info_row['h_cdr2_end']))
    else:
        cdr_ranges['h_cdr2'] = None
    
    if pd.notna(cdr_info_row.get('h_cdr3_start')) and pd.notna(cdr_info_row.get('h_cdr3_end')):
        cdr_ranges['h_cdr3'] = (int(cdr_info_row['h_cdr3_start']), int(cdr_info_row['h_cdr3_end']))
    else:
        cdr_ranges['h_cdr3'] = None
    
    # Light chain CDRs (optional for nanobodies)
    if pd.notna(cdr_info_row.get('l_cdr1_start')) and pd.notna(cdr_info_row.get('l_cdr1_end')):
        cdr_ranges['l_cdr1'] = (int(cdr_info_row['l_cdr1_start']), int(cdr_info_row['l_cdr1_end']))
    else:
        cdr_ranges['l_cdr1'] = None
    
    if pd.notna(cdr_info_row.get('l_cdr2_start')) and pd.notna(cdr_info_row.get('l_cdr2_end')):
        cdr_ranges['l_cdr2'] = (int(cdr_info_row['l_cdr2_start']), int(cdr_info_row['l_cdr2_end']))
    else:
        cdr_ranges['l_cdr2'] = None
    
    if pd.notna(cdr_info_row.get('l_cdr3_start')) and pd.notna(cdr_info_row.get('l_cdr3_end')):
        cdr_ranges['l_cdr3'] = (int(cdr_info_row['l_cdr3_start']), int(cdr_info_row['l_cdr3_end']))
    else:
        cdr_ranges['l_cdr3'] = None
    
    return cdr_ranges


def get_cdr_residue_set(cdr_ranges: Dict[str, Optional[Tuple[int, int]]], 
                        chain_id: str, 
                        sequence_length: int) -> Set[int]:
    """
    Get set of residue indices (1-based) that are in CDR regions for a given chain.
    
    Args:
        cdr_ranges: Dictionary of CDR ranges from parse_cdr_ranges
        chain_id: Chain ID ('H' for heavy, 'L' for light)
        sequence_length: Length of the chain sequence
        
    Returns:
        Set of residue indices (1-based) that are in CDR regions
    """
    cdr_residues = set()
    
    if chain_id.upper() == 'H':
        # Heavy chain CDRs
        for cdr_name in ['h_cdr1', 'h_cdr2', 'h_cdr3']:
            if cdr_ranges.get(cdr_name) is not None:
                start, end = cdr_ranges[cdr_name]
                # CDR ranges are inclusive
                for res_idx in range(start, end + 1):
                    if 1 <= res_idx <= sequence_length:
                        cdr_residues.add(res_idx)
    elif chain_id.upper() == 'L':
        # Light chain CDRs
        for cdr_name in ['l_cdr1', 'l_cdr2', 'l_cdr3']:
            if cdr_ranges.get(cdr_name) is not None:
                start, end = cdr_ranges[cdr_name]
                # CDR ranges are inclusive
                for res_idx in range(start, end + 1):
                    if 1 <= res_idx <= sequence_length:
                        cdr_residues.add(res_idx)
    
    return cdr_residues


def get_cdr_residue_set_from_pdb(
    cdr_ranges: Dict[str, Optional[Tuple[int, int]]],
    chain,
    chain_type: str,  # 'H' or 'L'
) -> Set[Tuple]:
    """
    Get set of (chain_id, res_id) tuples for residues in CDR regions.
    Uses PDB residue numbering from the chain.
    """
    cdr_set = set()
    cdr_names = ['h_cdr1', 'h_cdr2', 'h_cdr3'] if chain_type == 'H' else ['l_cdr1', 'l_cdr2', 'l_cdr3']
    for cdr_name in cdr_names:
        rng = cdr_ranges.get(cdr_name)
        if rng is None:
            continue
        start, end = rng
        for res in chain.get_residues():
            res_id = res.get_id()
            pdb_num = res_id[1] if isinstance(res_id, tuple) else res_id
            if start <= pdb_num <= end:
                cdr_set.add((chain.id, res_id))
    return cdr_set


def calculate_fixed_residues_for_antibody(
    pdb_path: Path,
    cdr_info_row: pd.Series,
    heavy_chain_id: Optional[str] = None,
    light_chain_id: Optional[str] = None,
) -> List[str]:
    """
    Calculate fixed residues (scaffold) for antibody inverse folding.
    
    Fixed residues:
    - H chain: All residues EXCEPT CDR loops
    - L chain: All residues EXCEPT CDR loops (if present; nanobody has no L)
    - All other chains: ALL residues (antigen, etc.)
    
    Chain IDs come from cdr_info_row['h_chain'] and ['l_chain'].
    Fallback to 'H' and 'L' if not provided (backward compatibility).
    
    Args:
        pdb_path: Path to PDB/CIF file
        cdr_info_row: Pandas Series with CDR information (must include h_chain, l_chain for correct chain mapping)
        heavy_chain_id: Override for H chain ID (if None, use cdr_info_row['h_chain'] or 'H')
        light_chain_id: Override for L chain ID (if None, use cdr_info_row['l_chain'] or 'L'; empty for nanobody)
        
    Returns:
        List of fixed residue identifiers in format "A123", "B456" (using actual chain IDs from structure)
    """
    # Get chain IDs from CSV (required: design models must provide h_chain and l_chain)
    h_chain = heavy_chain_id
    if h_chain is None:
        val = cdr_info_row.get('h_chain')
        if pd.isna(val) or not str(val).strip():
            raise ValueError(
                "h_chain is required in CDR info CSV. "
                "Design models must provide the heavy chain ID in the structure."
            )
        h_chain = str(val).strip()
    
    l_chain = light_chain_id
    if l_chain is None:
        val = cdr_info_row.get('l_chain')
        if pd.notna(val) and str(val).strip() and str(val).strip() not in ('', 'nan'):
            l_chain = str(val).strip()
        else:
            l_chain = None  # Nanobody: no light chain
    
    # Parse structure
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('antibody', str(pdb_path))
    except Exception:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('antibody', str(pdb_path))
    
    cdr_ranges = parse_cdr_ranges(cdr_info_row)
    chains = list(structure.get_chains())
    
    heavy_chain = None
    light_chain = None
    other_chains = []
    
    for chain in chains:
        if chain.id == h_chain:
            heavy_chain = chain
        elif l_chain and chain.id == l_chain:
            light_chain = chain
        else:
            other_chains.append(chain)
    
    if heavy_chain is None:
        raise ValueError(f"Heavy chain '{h_chain}' not found in structure. Available chains: {[c.id for c in chains]}")
    
    fixed_residues = []
    
    # H chain: fix non-CDR
    heavy_cdr_set = get_cdr_residue_set_from_pdb(cdr_ranges, heavy_chain, 'H')
    for res in heavy_chain.get_residues():
        res_id = res.get_id()
        if (h_chain, res_id) not in heavy_cdr_set:
            pdb_res_num = res_id[1] if isinstance(res_id, tuple) else res_id
            fixed_residues.append(f"{h_chain}{pdb_res_num}")
    
    # L chain: fix non-CDR (if present)
    if light_chain is not None:
        light_cdr_set = get_cdr_residue_set_from_pdb(cdr_ranges, light_chain, 'L')
        for res in light_chain.get_residues():
            res_id = res.get_id()
            if (l_chain, res_id) not in light_cdr_set:
                pdb_res_num = res_id[1] if isinstance(res_id, tuple) else res_id
                fixed_residues.append(f"{l_chain}{pdb_res_num}")
    
    # All other chains: fix ALL residues (antigen, etc.)
    for chain in other_chains:
        for res in chain.get_residues():
            res_id = res.get_id()
            pdb_res_num = res_id[1] if isinstance(res_id, tuple) else res_id
            fixed_residues.append(f"{chain.id}{pdb_res_num}")
    
    return fixed_residues


def load_cdr_info_csv(cdr_info_csv: str) -> pd.DataFrame:
    """
    Load and validate CDR info CSV.
    
    Required columns:
    - id: Identifier for the antibody (should match PDB filename)
    - h_chain: Chain ID of heavy chain in the structure (required)
    - l_chain: Chain ID of light chain in the structure (required for antibody, empty for nanobody)
    - h_cdr1_start, h_cdr1_end, h_cdr2_start, h_cdr2_end, h_cdr3_start, h_cdr3_end
    - l_cdr1_start, l_cdr1_end, l_cdr2_start, l_cdr2_end, l_cdr3_start, l_cdr3_end (optional for nanobodies)
    
    Args:
        cdr_info_csv: Path to CDR info CSV file
        
    Returns:
        DataFrame with CDR information
    """
    df = pd.read_csv(cdr_info_csv)
    
    # Validate required columns
    required_cols = ['id', 'h_chain', 'l_chain', 'h_cdr1_start', 'h_cdr1_end', 'h_cdr2_start', 'h_cdr2_end',
                     'h_cdr3_start', 'h_cdr3_end']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CDR info CSV: {missing_cols}")
    
    return df


def match_pdb_to_cdr_info(pdb_path: Path, cdr_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Match a PDB file to its CDR information row.

    Matching strategy (supports both per-design and per-target CDR CSV):
    1. Try exact match on full design id (e.g. "01_7UXQ_0" for 01_7UXQ_0.cif)
    2. Fallback to target name (e.g. "01_7UXQ") for per-target CSV
    3. Support "target:scaffold" format in CSV

    Args:
        pdb_path: Path to PDB file
        cdr_df: DataFrame with CDR information

    Returns:
        Series with CDR information for this PDB, or None if not found
    """
    import re

    pdb_stem = pdb_path.stem  # e.g. 01_7UXQ_0

    # Pattern: {sequence_number}_{target_id}_{index} or {sequence_number}_{target_id}
    target_pattern = r'^(\d{2}_[A-Z0-9]{4,})'
    match = re.match(target_pattern, pdb_stem)
    if match:
        target_name = match.group(1)
    else:
        base_name = re.sub(r'[-_]?\d+$', '', pdb_stem)
        target_name = base_name

    # 1. Try exact match on full design id (per-design: id = 01_7UXQ_0)
    matches = cdr_df[cdr_df['id'] == pdb_stem]
    if len(matches) > 0:
        return matches.iloc[0]

    # 2. Fallback: per-target (id = 01_7UXQ)
    matches = cdr_df[cdr_df['id'] == target_name]
    if len(matches) > 0:
        return matches.iloc[0]
    
    # 3. Legacy "target:scaffold" format (do not use startswith(target_name) - would wrongly match 01_7UXQ_0 for 01_7UXQ_99)
    for idx, row in cdr_df.iterrows():
        csv_id = str(row['id'])
        if csv_id.replace(':', '-') == target_name or csv_id.replace(':', '_') == target_name:
            return row
        if ':' in csv_id:
            csv_base = csv_id.split(':')[0] + '_' + csv_id.split(':')[1].split('_')[0] if '_' in csv_id.split(':')[1] else csv_id.split(':')[0]
            if csv_base == target_name:
                return row

    return None
