"""
Utilities for processing AME CSV input files.

CSV format:
- Column 1: ID (corresponds to structure filename)
- Column 2: Task name (one of the 41 AME tasks, e.g., "M0024_1nzy")
- Column 3: Motif residues (comma-separated, e.g., "A22,A48")
"""
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# Valid AME task names (from reference_pdbs directory)
VALID_AME_TASKS = {
    "M0024_1nzy", "M0040_13pk", "M0050_1dbt", "M0054_1qfe", "M0058_1cju",
    "M0078_1al6", "M0092_1dli", "M0093_1dqa", "M0096_1chm", "M0097_1ctt",
    "M0110_1c0p", "M0129_1os7", "M0151_1q0n", "M0157_1qh5", "M0179_1q3s",
    "M0188_1xel", "M0209_1lij", "M0255_1mg5", "M0315_1ey3", "M0349_1e3v",
    "M0365_1pfk", "M0375_4ts9", "M0500_1e3i", "M0552_1fgh", "M0555_1f8r",
    "M0584_1ldm", "M0630_1j79", "M0636_1uaq", "M0663_1rk2", "M0664_2dhn",
    "M0674_1uf7", "M0710_1ra0", "M0711_2esd", "M0717_1x7d", "M0731_1mt5",
    "M0732_1xs1", "M0738_1o98", "M0739_1knp", "M0870_1oh9", "M0904_1qgx",
    "M0907_1rbl"
}


def load_ame_csv(csv_path: str) -> pd.DataFrame:
    """
    Load AME CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with columns: ['id', 'task', 'motif_residues']
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"AME CSV file not found: {csv_path}")
    
    # Try to read CSV (handle both with and without header)
    # Note: motif_residues field contains commas, so we need to handle it specially
    try:
        # Read only first two columns, then manually parse the rest
        df = pd.read_csv(csv_path, header=None, usecols=[0, 1], names=['id', 'task'])
        
        # Manually read the full file to extract motif_residues (everything after second comma)
        motif_residues_list = []
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by comma, but only split on first two commas
                parts = line.split(',', 2)
                if len(parts) >= 3:
                    motif_residues_list.append(parts[2])
                elif len(parts) == 2:
                    motif_residues_list.append('')
                else:
                    motif_residues_list.append('')
        
        df['motif_residues'] = motif_residues_list
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")
    
    # Validate required columns exist
    if len(df.columns) < 3:
        raise ValueError(f"CSV must have at least 3 columns. Found {len(df.columns)} columns.")
    
    # Strip whitespace from string columns
    df['id'] = df['id'].astype(str).str.strip()
    df['task'] = df['task'].astype(str).str.strip()
    df['motif_residues'] = df['motif_residues'].astype(str).str.strip()
    
    # Validate task names
    invalid_tasks = df[~df['task'].isin(VALID_AME_TASKS)]['task'].unique()
    if len(invalid_tasks) > 0:
        print(f"Warning: Found {len(invalid_tasks)} invalid task names: {invalid_tasks}")
        print(f"Valid task names are: {sorted(VALID_AME_TASKS)}")
    
    return df


def parse_motif_residues(motif_str: str) -> List[str]:
    """
    Parse motif residues string into list of residue identifiers.
    
    Args:
        motif_str: Comma-separated string like "A22,A48" or "A22, A48" or '"A22,A48"'
        
    Returns:
        List of residue identifiers like ["A22", "A48"]
    """
    if pd.isna(motif_str) or motif_str == '':
        return []
    
    # Remove surrounding quotes if present (handles CSV quoted fields)
    motif_str = str(motif_str).strip()
    if motif_str.startswith('"') and motif_str.endswith('"'):
        motif_str = motif_str[1:-1]
    elif motif_str.startswith("'") and motif_str.endswith("'"):
        motif_str = motif_str[1:-1]
    
    # Split by comma and strip whitespace
    residues = [r.strip() for r in motif_str.split(',')]
    # Filter out empty strings
    residues = [r for r in residues if r]
    return residues


def match_pdb_to_csv_info(pdb_path: Path, csv_df: pd.DataFrame) -> Optional[Dict]:
    """
    Match a PDB file to its CSV row information.
    
    Args:
        pdb_path: Path to PDB file (may be in formatted_designs/ with renamed filename)
        csv_df: DataFrame from load_ame_csv()
        
    Returns:
        Dictionary with keys: 'id', 'task', 'motif_residues' (as list), or None if not found
    """
    # Try to match by filename (with or without extension)
    pdb_name = pdb_path.name
    pdb_stem = pdb_path.stem
    
    # Try exact match first (with extension)
    row = csv_df[csv_df['id'] == pdb_name]
    if len(row) == 0:
        # Try without extension
        row = csv_df[csv_df['id'] == pdb_stem]
    
    # If still no match, try partial matching (for preprocessed files like "ame_input-1.pdb")
    # Match if CSV ID is contained in filename or vice versa
    if len(row) == 0:
        for idx, csv_row in csv_df.iterrows():
            csv_id = str(csv_row['id']).strip()
            csv_id_no_ext = Path(csv_id).stem
            
            # Check if CSV ID (or its stem) is contained in PDB filename
            if csv_id in pdb_stem or csv_id_no_ext in pdb_stem:
                row = csv_df.iloc[[idx]]
                break
            
            # Check if PDB stem is contained in CSV ID
            if pdb_stem in csv_id or pdb_stem in csv_id_no_ext:
                row = csv_df.iloc[[idx]]
                break
    
    # Fallback: If filename is in test_input-* format and CSV has only one entry, use it
    # This handles the case where preprocessing renamed files to test_input-1.pdb, test_input-1-1.pdb, etc.
    if len(row) == 0:
        import re
        # Check if filename matches test_input-* pattern
        if re.match(r'test_input-\d+(-\d+)?', pdb_stem):
            # If CSV has only one entry, use it
            if len(csv_df) == 1:
                row = csv_df.iloc[[0]]
    
    if len(row) == 0:
        return None
    
    # Take first match if multiple
    row = row.iloc[0]
    
    return {
        'id': row['id'],
        'task': row['task'],
        'motif_residues': parse_motif_residues(row['motif_residues'])
    }


def calculate_fixed_residues_from_motif(pdb_path: Path, motif_residues: List[str]) -> List[str]:
    """
    Calculate fixed residues from motif residues.
    
    In AME, we need to:
    - FIX motif residues (keep them unchanged)
    - REDESIGN all other protein residues
    
    For LigandMPNN's --fixed_residues_multi input,
    we need to provide the residues that should be FIXED (not designed).
    So we return only the motif residues as fixed residues.
    
    Args:
        pdb_path: Path to PDB file
        motif_residues: List of motif residue identifiers like ["A22", "A48"]
        
    Returns:
        List of fixed residue identifiers (only motif residues) like ["A22", "A48"]
    """
    from biotite.structure.io import pdb, pdbx
    
    # Read structure to validate motif residues exist
    if pdb_path.suffix.lower() == '.pdb':
        atom_array = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1)
    elif pdb_path.suffix.lower() in ['.cif', '.mmcif']:
        cif_file = pdbx.CIFFile.read(str(pdb_path))
        atom_array = pdbx.get_structure(cif_file, model=1)
    else:
        raise ValueError(f"Unsupported file format: {pdb_path.suffix}")
    
    # Filter to protein atoms only (exclude HETATM/ligand)
    # In biotite, hetero=True means HETATM, hetero=False means ATOM (protein)
    protein_atoms = atom_array[~atom_array.hetero]
    
    # Get unique protein residues to validate motif residues
    # Use CA atoms to get unique residues (same approach as inversefold_api.py)
    from biotite.structure import filter_amino_acids
    ca_atoms = protein_atoms[(protein_atoms.atom_name == 'CA')]
    
    protein_residues = set()
    for chain_id, res_id in zip(ca_atoms.chain_id, ca_atoms.res_id):
        # Handle res_id format (could be int or tuple)
        # For tuple format: (insertion_code_index, pdb_residue_number)
        # For int format: pdb_residue_number
        if isinstance(res_id, tuple):
            pdb_res_id = res_id[1] if len(res_id) > 1 else res_id[0]
        else:
            pdb_res_id = res_id
        # Format: chain_id + residue_number (e.g., "A22")
        residue_id = f"{chain_id}{pdb_res_id}"
        protein_residues.add(residue_id)
    
    # Parse motif residues into set for validation
    motif_set = set(motif_residues)
    
    # Validate that all motif residues exist in the structure
    missing_motif = motif_set - protein_residues
    if missing_motif:
        print(f"Warning: Some motif residues not found in structure: {sorted(missing_motif)}")
        print(f"Available protein residues: {sorted(protein_residues)[:20]}...")
    
    # Fixed residues = ONLY motif residues (all other protein residues will be redesigned)
    # Filter to only include motif residues that actually exist in the structure
    valid_fixed_residues = sorted(motif_set & protein_residues)
    
    if len(valid_fixed_residues) == 0:
        raise ValueError(f"No valid motif residues found in structure. Motif residues: {motif_residues}")
    
    return valid_fixed_residues


def print_ame_summary(csv_df: pd.DataFrame, input_dir: Path) -> None:
    """
    Print summary of AME analysis.
    
    Args:
        csv_df: DataFrame from load_ame_csv()
        input_dir: Input directory containing structure files
    """
    print("\n" + "="*80)
    print("AME Pipeline Summary")
    print("="*80)
    
    # Count total structures
    total_structures = len(csv_df)
    print(f"\nTotal structures analyzed: {total_structures}")
    
    # Count unique tasks
    unique_tasks = csv_df['task'].unique()
    num_tasks = len(unique_tasks)
    print(f"Number of unique tasks: {num_tasks} (out of 41 possible AME tasks)")
    
    # Count structures per task
    task_counts = csv_df['task'].value_counts().sort_index()
    print(f"\nStructures per task:")
    for task, count in task_counts.items():
        print(f"  {task}: {count} structure(s)")
    
    # Check for missing structures
    structure_files = set()
    for ext in ['.pdb', '.cif']:
        structure_files.update([f.stem for f in input_dir.rglob(f"*{ext}")])
    
    csv_ids = set(csv_df['id'].str.replace(r'\.(pdb|cif)$', '', regex=True))
    missing_in_csv = structure_files - csv_ids
    missing_files = csv_ids - structure_files
    
    if missing_in_csv:
        print(f"\nWarning: {len(missing_in_csv)} structure file(s) found in directory but not in CSV:")
        for f in sorted(list(missing_in_csv))[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_in_csv) > 10:
            print(f"  ... and {len(missing_in_csv) - 10} more")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} CSV entry(ies) have no corresponding structure file:")
        for f in sorted(list(missing_files))[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    print("="*80 + "\n")
