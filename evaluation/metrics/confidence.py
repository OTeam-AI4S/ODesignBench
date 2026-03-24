import json
import os
import pickle
from biopandas.pdb import PandasPdb
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdb, pdbx
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Set

def calculate_ipae_info(pae_matrix: np.ndarray, chain_indices: np.ndarray) -> dict:
    """
    Calculate iPAE (interface PAE) related information from complete PAE matrix.

    Args:
        pae_matrix (np.ndarray): N x N PAE matrix, where N is total number of residues.
        chain_indices (np.ndarray): Array of length N, indicating chain index for each residue (0, 1, 2...).

    Returns:
        dict: Dictionary containing iPAE statistics:
              - 'mean_ipae': Mean of all inter-chain PAE values.
              - 'min_ipae': Minimum of all inter-chain PAE values.
              - 'ipae_values': 1D array of all inter-chain PAE values.
              - 'ipae_mask': N x N boolean mask, True indicates inter-chain positions.
              - 'ipae_blocks': Dictionary storing iPAE submatrices for each chain pair.
    """
    num_residues = pae_matrix.shape[0]
    if num_residues != len(chain_indices):
        raise ValueError("PAE matrix dimensions do not match chain indices length.")

    # --- Core calculation: Use broadcasting to create inter-chain mask ---
    # Convert chain indices to column and row vectors
    chain_col = chain_indices[:, np.newaxis]
    chain_row = chain_indices[np.newaxis, :]
    
    # Mask is True when two residues have different chain indices
    ipae_mask = (chain_col != chain_row)
    
    # Extract all iPAE values using mask
    ipae_values = pae_matrix[ipae_mask]
    
    if ipae_values.size == 0:
        # If only one chain, no iPAE values
        return {
            'mean_ipae': np.nan, 'min_ipae': np.nan,
            'ipae_values': np.array([]), 'ipae_mask': ipae_mask,
            'ipae_blocks': {}
        }

    # Calculate key statistics
    mean_ipae = ipae_values.mean()
    min_ipae = ipae_values.min()

    # Extract iPAE submatrices for each chain pair
    ipae_blocks = {}
    unique_chains = np.unique(chain_indices)
    for i in range(len(unique_chains)):
        for j in range(i + 1, len(unique_chains)):
            chain_id_1 = unique_chains[i]
            chain_id_2 = unique_chains[j]
            
            mask_chain_1 = (chain_indices == chain_id_1)
            mask_chain_2 = (chain_indices == chain_id_2)
            
            # Extract A-B and B-A iPAE blocks
            block_ab = pae_matrix[mask_chain_1, :][:, mask_chain_2]
            block_ba = pae_matrix[mask_chain_2, :][:, mask_chain_1]

            ipae_blocks[f'chain_{chain_id_1}-chain_{chain_id_2}'] = block_ab
            ipae_blocks[f'chain_{chain_id_2}-chain_{chain_id_1}'] = block_ba


    return {
        'mean_ipae': mean_ipae,
        'min_ipae': min_ipae,
        'ipae_values': ipae_values,
        'ipae_mask': ipae_mask,
        'ipae_blocks': ipae_blocks
    }

def _unique_in_order(values: Sequence) -> List:
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out

def _parse_chain_list_csv(val) -> List[str]:
    """
    Parse chain list strings like "A,B" (or "A, B") into ["A","B"].
    Empty/NaN -> [].
    """
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def _chain_order_from_atom_array(atom_array) -> List[str]:
    # Preserve appearance order; np.unique() sorts which breaks mapping.
    return _unique_in_order([str(x) for x in atom_array.chain_id.tolist()])

def _calc_cross_pae_stats(
    pae_matrix: np.ndarray,
    token_chain_ids: np.ndarray,
    ab_token_chain_ids: Set[str],
    ag_token_chain_ids: Set[str],
) -> Tuple[float, float]:
    """
    Compute mean/min PAE for Ab->Ag only:
      res_i in Ab (H+L), res_j in Ag (A,B,...)  (directional).
    token_chain_ids may be numeric or string-like; comparisons are label-based.
    """
    if pae_matrix.shape[0] != token_chain_ids.shape[0]:
        raise ValueError("PAE matrix dimensions do not match token_chain_ids length.")
    if not ab_token_chain_ids or not ag_token_chain_ids:
        return (float("nan"), float("nan"))
    mask_i = np.isin(token_chain_ids, list(ab_token_chain_ids))
    mask_j = np.isin(token_chain_ids, list(ag_token_chain_ids))
    cross_mask = mask_i[:, None] & mask_j[None, :]
    vals = pae_matrix[cross_mask]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(vals.mean()), float(vals.min()))

def letter_to_number(char: str):
    """
    Convert a single English letter to its corresponding number in the alphabet (A=1, B=2, ...).

    Args:
        char: A single character string.

    Returns:
        If input is an English letter, returns integer between 1-26.
        Otherwise returns None.
    """
    # Check if input is a single character string
    if not isinstance(char, str) or len(char) != 1:
        return None
    
    # Convert to uppercase for case-insensitive comparison
    char_upper = char.upper()
    
    # Check if it's an English letter
    if 'A' <= char_upper <= 'Z':
        # Calculate using ASCII code
        return ord(char_upper) - ord('A') + 1
    else:
        return None

class Confidence:

    def __init__(self):
        pass
    
    @staticmethod
    def gather_af3_confidence(
        confidence_path: str,
        summary_confidence_path: str,
        pdbpath: str,
        h_chain_id: Optional[str] = None,
        l_chain_id: Optional[str] = None,
        ag_chain_ids: Optional[Sequence[str]] = None,
    ):
        with open(summary_confidence_path, "r") as f:
            summary_confidence = json.load(f)
        with open(confidence_path, "r") as f:
            confidence = json.load(f)
        
        iptm = summary_confidence['iptm']
        pae = np.array(confidence['pae'])
        token_chain_ids_raw = np.array(confidence['token_chain_ids'])
        # AF3 outputs may use numeric or string chain labels (e.g., 0/1 or A/B).
        # Normalize to stripped strings so downstream matching is type-robust.
        token_chain_ids = np.array([str(x).strip() for x in token_chain_ids_raw.tolist()], dtype=object)
        pdb_file = pdb.PDBFile.read(pdbpath)
        atom_array = pdb.get_structure(pdb_file, model=1, extra_fields=['b_factor'])
        chain_list = _chain_order_from_atom_array(atom_array)

        # Map AF3 token_chain_ids to chain IDs in structure by assuming same chain order.
        token_chain_order = _unique_in_order(token_chain_ids.tolist())
        chain_to_token: Dict[str, str] = {}
        if len(token_chain_order) == len(chain_list):
            chain_to_token = {chain_list[i]: str(token_chain_order[i]).strip() for i in range(len(chain_list))}

        # iPAE: default = all inter-chain, but for antibody tasks compute Ab->Ag only.
        ab_chains: List[str] = []
        if h_chain_id is not None and str(h_chain_id).strip():
            ab_chains.append(str(h_chain_id).strip())
        if l_chain_id is not None and str(l_chain_id).strip() and str(l_chain_id).strip().lower() != "nan":
            ab_chains.append(str(l_chain_id).strip())
        ag_chains = []
        if ag_chain_ids is not None:
            ag_chains = [str(x).strip() for x in ag_chain_ids if str(x).strip() and str(x).strip().lower() != "nan"]

        if chain_to_token and ab_chains and ag_chains:
            ab_token = {chain_to_token[c] for c in ab_chains if c in chain_to_token}
            ag_token = {chain_to_token[c] for c in ag_chains if c in chain_to_token}
            ipae, min_ipae = _calc_cross_pae_stats(pae, token_chain_ids, ab_token, ag_token)
        else:
            ipae_info = calculate_ipae_info(pae, token_chain_ids)
            ipae, min_ipae = ipae_info['mean_ipae'], ipae_info['min_ipae']

        # Binder pTM: use explicit H/L chain IDs when provided; otherwise fallback to legacy b_factor heuristic.
        chain_ptm = summary_confidence.get('chain_ptm', [])
        ptm_H = float("nan")
        ptm_L = float("nan")
        ptm_binder = float("nan")

        def _chain_len(chain_id: str) -> int:
            mask = atom_array.chain_id == chain_id
            if not np.any(mask):
                return 0
            return int(len(np.unique(atom_array.res_id[mask])))

        if chain_ptm and chain_list:
            # Use explicit H/L mapping if available.
            if h_chain_id and str(h_chain_id).strip() in chain_list:
                idx = chain_list.index(str(h_chain_id).strip())
                if idx < len(chain_ptm):
                    ptm_H = float(chain_ptm[idx])
            if l_chain_id and str(l_chain_id).strip() in chain_list:
                idx = chain_list.index(str(l_chain_id).strip())
                if idx < len(chain_ptm):
                    ptm_L = float(chain_ptm[idx])

            # If we got at least one of H/L, compute length-weighted binder pTM.
            if not np.isnan(ptm_H) or not np.isnan(ptm_L):
                wH = _chain_len(str(h_chain_id).strip()) if (h_chain_id and str(h_chain_id).strip()) else 0
                wL = _chain_len(str(l_chain_id).strip()) if (l_chain_id and str(l_chain_id).strip()) else 0
                num = 0.0
                den = 0.0
                if not np.isnan(ptm_H) and wH > 0:
                    num += ptm_H * wH
                    den += wH
                if not np.isnan(ptm_L) and wL > 0:
                    num += ptm_L * wL
                    den += wL
                if den > 0:
                    ptm_binder = num / den
                else:
                    # Fallback: simple mean of available chains
                    vals = [x for x in [ptm_H, ptm_L] if not np.isnan(x)]
                    ptm_binder = float(np.mean(vals)) if vals else float("nan")
            else:
                # Legacy heuristic: first chain with b_factor != 0 (often "designed" chain)
                designed = atom_array.b_factor != 0
                if np.any(designed):
                    chains_to_design = str(atom_array.chain_id[designed][0])
                else:
                    chains_to_design = str(chain_list[0])
                if chains_to_design in chain_list:
                    idx = chain_list.index(chains_to_design)
                    ptm_binder = float(chain_ptm[idx]) if idx < len(chain_ptm) else float(chain_ptm[0])
                else:
                    ptm_binder = float(chain_ptm[0]) if len(chain_ptm) > 0 else float("nan")

        plddt = sum(confidence['atom_plddts']) / len(confidence['atom_plddts'])
        # AF-style confidence files may store atom pLDDT either in 0-1 or 0-100 scale.
        # Normalize conservatively so downstream success thresholds remain meaningful.
        if plddt > 1.5:
            plddt = plddt / 100.0
        return plddt, ipae, min_ipae, iptm, ptm_binder, ptm_H, ptm_L
    
    @staticmethod
    def gather_chai1_confidence(cand: str, inverse_fold_path: str):
        # Check if cand has required attributes
        if not hasattr(cand, 'token_asym_id') or cand.token_asym_id is None:
            # Fallback: try to extract plddt from CIF file if available
            plddt = np.nan
            ipae = np.nan
            min_ipae = np.nan
            iptm = np.nan
            ptm_binder = np.nan
            
            # Try to get plddt from CIF file if cand has cif_paths
            if hasattr(cand, 'cif_paths') and cand.cif_paths:
                try:
                    from biotite.structure.io import pdbx
                    cif_file = pdbx.CIFFile.read(str(cand.cif_paths[0]))
                    atom_array = pdbx.get_structure(cif_file, model=1, extra_fields=['b_factor'])
                    if 'b_factor' in atom_array.get_annotation():
                        plddt_values = atom_array.b_factor[atom_array.atom_name == 'CA']
                        if len(plddt_values) > 0:
                            plddt = np.mean(plddt_values) / 100.0  # Convert from 0-100 scale to 0-1
                except Exception:
                    pass
            
            return plddt, ipae, min_ipae, iptm, ptm_binder
        
        token_asym_id = cand.token_asym_id.numpy()
        token_asym_id = token_asym_id[token_asym_id != 0]
        plddt = np.mean(cand.plddt.squeeze(0).numpy())
        pae = cand.pae.squeeze(0).numpy()
        ipae_info = calculate_ipae_info(pae, token_asym_id)
        ipae, min_ipae = ipae_info['mean_ipae'], ipae_info['min_ipae']
        iptm = cand.ranking_data[0].ptm_scores.interface_ptm.numpy()
        
        # Try to read PDB file for ptm_binder calculation
        ptm_binder = np.nan
        if inverse_fold_path and os.path.exists(inverse_fold_path):
            try:
                # trb = pickle.load(open(trb, 'rb'))
                atom_array = pdb.get_structure(pdb.PDBFile.read(inverse_fold_path), model=1, extra_fields=['b_factor'])
                chains_to_design = str(atom_array.chain_id[atom_array.b_factor == 0][0])
                binder_id = letter_to_number(chains_to_design)
                if binder_id is not None and binder_id > 0:
                    ptm_binder = cand.ranking_data[0].ptm_scores.per_chain_ptm[0, binder_id - 1].numpy()
            except (FileNotFoundError, IndexError, ValueError, KeyError) as e:
                # If PDB file doesn't exist or parsing fails, ptm_binder remains NaN
                pass
        
        return plddt, ipae, min_ipae, iptm, ptm_binder
    
    @staticmethod
    def gather_esmfold_confidence(pdb_path, chain_id=None):
        """
        Extract pLDDT values from B-factor column of a PDB file using PandasPdb
        
        Parameters:
        -----------
        pdb_path : str
            Path to the PDB file
        chain_id : str, optional
            Chain ID to filter for. If None, returns data for all chains
            
        Returns:
        --------
        pandas.DataFrame or pandas.Series
            B-factor values (pLDDT) for the specified chain or all chains
        """
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_path)
        
        # Get only CA atoms to avoid duplicate values per residue
        ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
        
        if chain_id:
            # breakpoint()
            # Filter for specific chain
            chain_data = ca_atoms[ca_atoms['chain_id'] == chain_id]
            return chain_data['b_factor'].mean()
        else:
            # Return data for all chains
            return ca_atoms['b_factor'].mean()
