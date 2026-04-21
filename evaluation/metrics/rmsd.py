import pickle
import re
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx, pdb
from biotite.structure import AtomArray, AtomArrayStack
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

class RMSDCalculator():
    def __init__(self):
        
        print(
            " RMSD calculator initialized. \
              benchmark      RMSD_method \
              AME            compute_atomic_motif_rmsd \
              MotifBench     compute_protein_ca_rmsd + compute_motif_rmsd \
              FreeNucleotide compute_C4_rmsd \
              FreeProtein    compute_protein_backbone_rmsd \
              ProteinBinder  compute_protein_backbone_rmsd \
              PocketBench    compute_pocket_rmsd + compute_protein_backbone_rmsd \
              "
              )

    @staticmethod
    def _load_structure(path: str):
        if path.endswith('.cif'):
            return pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
        return pdb.get_structure(pdb.PDBFile.read(path), model=1)

    @staticmethod
    def _normalize_chain_ids(chain_ids: Optional[Sequence[str]]) -> Optional[Set[str]]:
        if chain_ids is None:
            return None
        normalized = {
            str(c).strip()
            for c in chain_ids
            if str(c).strip() and str(c).strip().lower() != "nan"
        }
        return normalized or None

    @staticmethod
    def _ca_coord_map(arr, chain_ids: Optional[Sequence[str]] = None) -> Dict[Tuple[str, int], np.ndarray]:
        chain_set = RMSDCalculator._normalize_chain_ids(chain_ids)
        mask = (arr.atom_name == "CA") & (~arr.hetero)
        if chain_set is not None:
            mask &= np.isin(arr.chain_id, list(chain_set))
        out: Dict[Tuple[str, int], np.ndarray] = {}
        if np.sum(mask) == 0:
            return out
        for chain_id, res_id, coord in zip(arr.chain_id[mask], arr.res_id[mask], arr.coord[mask]):
            key = (str(chain_id), int(res_id))
            if key not in out:
                out[key] = coord
        return out

    @staticmethod
    def _coords_from_res_id_map(coord_map: Dict[Tuple[str, int], np.ndarray]) -> Dict[int, np.ndarray]:
        by_res_id: Dict[int, np.ndarray] = {}
        for (_, res_id), coord in coord_map.items():
            if res_id not in by_res_id:
                by_res_id[res_id] = coord
        return by_res_id

    @staticmethod
    def _residue_atom_coord_map(
        arr,
        atom_name: str,
        chain_ids: Optional[Sequence[str]] = None,
        hetero: Optional[bool] = False,
    ) -> Dict[Tuple[str, int, str], np.ndarray]:
        chain_set = RMSDCalculator._normalize_chain_ids(chain_ids)
        mask = arr.atom_name == atom_name
        if hetero is not None:
            mask &= arr.hetero == hetero
        if chain_set is not None:
            mask &= np.isin(arr.chain_id, list(chain_set))

        out: Dict[Tuple[str, int, str], np.ndarray] = {}
        if np.sum(mask) == 0:
            return out

        annotation_categories = set(arr.get_annotation_categories())
        has_ins_code = "ins_code" in annotation_categories
        for idx in np.flatnonzero(mask):
            ins_code = str(arr.ins_code[idx]).strip() if has_ins_code else ""
            key = (str(arr.chain_id[idx]), int(arr.res_id[idx]), ins_code)
            if key not in out:
                out[key] = arr.coord[idx]
        return out

    @staticmethod
    def _shared_coord_arrays(
        ref_coord_map: Dict[Tuple[str, int, str], np.ndarray],
        pred_coord_map: Dict[Tuple[str, int, str], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, str]]]:
        shared_keys = sorted(set(ref_coord_map) & set(pred_coord_map))
        if not shared_keys:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), []

        ref_coords = np.stack([ref_coord_map[key] for key in shared_keys], axis=0)
        pred_coords = np.stack([pred_coord_map[key] for key in shared_keys], axis=0)
        return ref_coords, pred_coords, shared_keys

    @staticmethod
    def _normalize_residue_keys(
        residue_keys: Optional[Sequence[str]],
    ) -> Optional[Set[str]]:
        if residue_keys is None:
            return None
        normalized = {
            str(key).strip()
            for key in residue_keys
            if str(key).strip() and str(key).strip().lower() != "nan"
        }
        return normalized or None

    @staticmethod
    def _backbone_atom_coord_map(
        arr,
        residue_keys: Optional[Sequence[str]] = None,
        chain_ids: Optional[Sequence[str]] = None,
    ) -> Dict[Tuple[str, int, str, str], np.ndarray]:
        residue_set = RMSDCalculator._normalize_residue_keys(residue_keys)
        chain_set = RMSDCalculator._normalize_chain_ids(chain_ids)
        mask = np.isin(arr.atom_name, ["N", "CA", "C", "O"]) & (~arr.hetero)
        if chain_set is not None:
            mask &= np.isin(arr.chain_id, list(chain_set))

        out: Dict[Tuple[str, int, str, str], np.ndarray] = {}
        if np.sum(mask) == 0:
            return out

        annotation_categories = set(arr.get_annotation_categories())
        has_ins_code = "ins_code" in annotation_categories
        for idx in np.flatnonzero(mask):
            residue_key = f"{str(arr.chain_id[idx])}{int(arr.res_id[idx])}"
            if residue_set is not None and residue_key not in residue_set:
                continue
            ins_code = str(arr.ins_code[idx]).strip() if has_ins_code else ""
            key = (
                str(arr.chain_id[idx]),
                int(arr.res_id[idx]),
                ins_code,
                str(arr.atom_name[idx]),
            )
            if key not in out:
                out[key] = arr.coord[idx]
        return out

    @staticmethod
    def _backbone_atom_coord_map_by_res_id(
        arr,
        residue_ids: Optional[Sequence[int]] = None,
    ) -> Dict[Tuple[int, str], np.ndarray]:
        residue_id_set = None
        if residue_ids is not None:
            residue_id_set = {int(x) for x in residue_ids}

        mask = np.isin(arr.atom_name, ["N", "CA", "C", "O"]) & (~arr.hetero)
        out: Dict[Tuple[int, str], np.ndarray] = {}
        if np.sum(mask) == 0:
            return out

        for idx in np.flatnonzero(mask):
            res_id = int(arr.res_id[idx])
            if residue_id_set is not None and res_id not in residue_id_set:
                continue
            key = (res_id, str(arr.atom_name[idx]))
            if key not in out:
                out[key] = arr.coord[idx]
        return out

    @staticmethod
    def _residue_ids_from_keys(residue_keys: Sequence[str]) -> List[int]:
        residue_ids: List[int] = []
        for key in residue_keys:
            match = re.search(r"(-?\d+)$", str(key).strip())
            if match is None:
                continue
            residue_ids.append(int(match.group(1)))
        return sorted(set(residue_ids))
    
    @staticmethod
    def compute_C4_rmsd(pred: str, refold: str):
        # Placeholder for RMSD calculation logic
        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)
        
        pred_c4_mask = (pred_structure.atom_name == "C4'")
        refold_c4_mask = (refold_structure.atom_name == "C4'")
        pred_coord_align, _ = struc.superimpose(refold_structure.coord[refold_c4_mask], pred_structure.coord[pred_c4_mask])
        c4_rmsd = struc.rmsd(refold_structure.coord[refold_c4_mask], pred_coord_align)
        return c4_rmsd
    
    @staticmethod
    def compute_protein_backbone_rmsd(pred: str, refold: str):
        pred_structure = RMSDCalculator._load_structure(pred)
        refold_structure = RMSDCalculator._load_structure(refold)

        pred_map = RMSDCalculator._backbone_atom_coord_map(pred_structure)
        ref_map = RMSDCalculator._backbone_atom_coord_map(refold_structure)
        common = sorted(set(pred_map.keys()) & set(ref_map.keys()))
        if len(common) >= 3:
            pred_coords = np.stack([pred_map[k] for k in common], axis=0)
            ref_coords = np.stack([ref_map[k] for k in common], axis=0)
            pred_align, _ = struc.superimpose(ref_coords, pred_coords)
            return float(struc.rmsd(ref_coords, pred_align))

        pred_by_res = RMSDCalculator._backbone_atom_coord_map_by_res_id(pred_structure)
        ref_by_res = RMSDCalculator._backbone_atom_coord_map_by_res_id(refold_structure)
        common_by_res = sorted(set(pred_by_res.keys()) & set(ref_by_res.keys()))
        if len(common_by_res) >= 3:
            pred_coords = np.stack([pred_by_res[k] for k in common_by_res], axis=0)
            ref_coords = np.stack([ref_by_res[k] for k in common_by_res], axis=0)
            pred_align, _ = struc.superimpose(ref_coords, pred_coords)
            return float(struc.rmsd(ref_coords, pred_align))

        pred_mask = np.isin(pred_structure.atom_name, ["N", "CA", "C", "O"]) & (~pred_structure.hetero)
        ref_mask = np.isin(refold_structure.atom_name, ["N", "CA", "C", "O"]) & (~refold_structure.hetero)
        pred_coords = pred_structure.coord[pred_mask]
        ref_coords = refold_structure.coord[ref_mask]
        if len(pred_coords) != len(ref_coords) or len(pred_coords) < 3:
            return float("nan")
        pred_align, _ = struc.superimpose(ref_coords, pred_coords)
        return float(struc.rmsd(ref_coords, pred_align))

    @staticmethod
    def compute_protein_backbone_rmsd_subset(
        pred: str,
        refold: str,
        residue_keys: Sequence[str],
    ):
        """
        Backbone RMSD on a subset of residues, aligned on the same subset.
        """
        residue_set = RMSDCalculator._normalize_residue_keys(residue_keys)
        if not residue_set:
            return float("nan")

        pred_structure = RMSDCalculator._load_structure(pred)
        refold_structure = RMSDCalculator._load_structure(refold)

        pred_map = RMSDCalculator._backbone_atom_coord_map(
            pred_structure,
            residue_keys=list(residue_set),
        )
        ref_map = RMSDCalculator._backbone_atom_coord_map(
            refold_structure,
            residue_keys=list(residue_set),
        )
        common = sorted(set(pred_map.keys()) & set(ref_map.keys()))
        if len(common) >= 3:
            pred_coords = np.stack([pred_map[k] for k in common], axis=0)
            ref_coords = np.stack([ref_map[k] for k in common], axis=0)
            pred_align, _ = struc.superimpose(ref_coords, pred_coords)
            return float(struc.rmsd(ref_coords, pred_align))

        residue_ids = RMSDCalculator._residue_ids_from_keys(list(residue_set))
        if not residue_ids:
            return float("nan")

        pred_by_res = RMSDCalculator._backbone_atom_coord_map_by_res_id(
            pred_structure,
            residue_ids=residue_ids,
        )
        ref_by_res = RMSDCalculator._backbone_atom_coord_map_by_res_id(
            refold_structure,
            residue_ids=residue_ids,
        )
        common_by_res = sorted(set(pred_by_res.keys()) & set(ref_by_res.keys()))
        if len(common_by_res) < 3:
            return float("nan")

        pred_coords = np.stack([pred_by_res[k] for k in common_by_res], axis=0)
        ref_coords = np.stack([ref_by_res[k] for k in common_by_res], axis=0)
        pred_align, _ = struc.superimpose(ref_coords, pred_coords)
        return float(struc.rmsd(ref_coords, pred_align))
    
    @staticmethod
    def compute_pocket_rmsd(pred: str, refold: str, trb: str):

        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)

        trb = pickle.load(open(trb, 'rb'))
        pocket_residues = np.unique(np.char.add(trb.chain_id[~trb.condition_token_mask], np.array(trb.res_id[~trb.condition_token_mask], dtype=str)))
        pred_structure_pocket_backbone = pred_structure[np.char.add(pred_structure.chain_id, np.array(pred_structure.res_id, dtype=str)).isin(pocket_residues) & (pred_structure.atom_name.isin(["N", "CA", "C", "O"]))]
        refold_structure_pocket_backbone = refold_structure[np.char.add(refold_structure.chain_id, np.array(refold_structure.res_id, dtype=str)).isin(pocket_residues) & (refold_structure.atom_name.isin(["N", "CA", "C", "O"]))]
        pred_coord_align, _ = struc.superimpose(refold_structure_pocket_backbone, pred_structure_pocket_backbone)
        pocket_rmsd = struc.rmsd(refold_structure_pocket_backbone.coord, pred_coord_align)
        return pocket_rmsd
    
    @staticmethod
    def compute_protein_ca_rmsd(pred: str, refold: str):
        pred_structure = RMSDCalculator._load_structure(pred)
        refold_structure = RMSDCalculator._load_structure(refold)

        pred_map = RMSDCalculator._ca_coord_map(pred_structure)
        ref_map = RMSDCalculator._ca_coord_map(refold_structure)
        common = sorted(set(pred_map.keys()) & set(ref_map.keys()))
        if len(common) < 3:
            return float("nan")

        pred_coords = np.stack([pred_map[k] for k in common], axis=0)
        ref_coords = np.stack([ref_map[k] for k in common], axis=0)
        pred_align, _ = struc.superimpose(ref_coords, pred_coords)
        return float(struc.rmsd(ref_coords, pred_align))

    @staticmethod
    def compute_protein_ca_rmsd_chain_subset(
        pred: str,
        refold: str,
        chain_ids: Sequence[str],
    ):
        """
        CA RMSD restricted to a subset of chains (e.g., antibody-only H+L).
        Robust to missing residues by intersecting (chain_id, res_id) keys.
        """
        chain_set = set([str(c).strip() for c in (chain_ids or []) if str(c).strip() and str(c).strip().lower() != "nan"])
        if not chain_set:
            return float("nan")

        pred_structure = RMSDCalculator._load_structure(pred)
        refold_structure = RMSDCalculator._load_structure(refold)

        pred_map = RMSDCalculator._ca_coord_map(pred_structure, chain_ids=list(chain_set))
        ref_map = RMSDCalculator._ca_coord_map(refold_structure, chain_ids=list(chain_set))
        common = sorted(set(pred_map.keys()) & set(ref_map.keys()))
        if len(common) < 3:
            return float("nan")

        pred_coords = np.stack([pred_map[k] for k in common], axis=0)
        ref_coords = np.stack([ref_map[k] for k in common], axis=0)
        pred_align, _ = struc.superimpose(ref_coords, pred_coords)
        return float(struc.rmsd(ref_coords, pred_align))

    @staticmethod
    def compute_protein_ca_rmsd_unbound_vs_bound_design(
        unbound: str,
        bound_complex: str,
        bound_chain_ids: Sequence[str],
    ):
        """
        Compare the AF3-predicted unbound binder monomer against the binder chain
        extracted from the AF3-predicted target+binder complex.

        Chain IDs may differ between the monomer and complex outputs, so the primary
        matching key is residue index within the selected chain(s), with an ordered
        fallback when residue IDs are unavailable/misaligned.
        """
        unbound_structure = RMSDCalculator._load_structure(unbound)
        bound_structure = RMSDCalculator._load_structure(bound_complex)

        unbound_map = RMSDCalculator._ca_coord_map(unbound_structure)
        bound_map = RMSDCalculator._ca_coord_map(bound_structure, chain_ids=bound_chain_ids)
        if len(unbound_map) < 3 or len(bound_map) < 3:
            return float("nan")

        unbound_by_res = RMSDCalculator._coords_from_res_id_map(unbound_map)
        bound_by_res = RMSDCalculator._coords_from_res_id_map(bound_map)
        common_res_ids = sorted(set(unbound_by_res.keys()) & set(bound_by_res.keys()))
        if len(common_res_ids) >= 3:
            unbound_coords = np.stack([unbound_by_res[k] for k in common_res_ids], axis=0)
            bound_coords = np.stack([bound_by_res[k] for k in common_res_ids], axis=0)
            unbound_align, _ = struc.superimpose(bound_coords, unbound_coords)
            return float(struc.rmsd(bound_coords, unbound_align))

        unbound_coords = np.stack(list(unbound_map.values()), axis=0)
        bound_coords = np.stack(list(bound_map.values()), axis=0)
        if len(unbound_coords) != len(bound_coords) or len(unbound_coords) < 3:
            return float("nan")

        unbound_align, _ = struc.superimpose(bound_coords, unbound_coords)
        return float(struc.rmsd(bound_coords, unbound_align))

    @staticmethod
    def compute_atomic_motif_rmsd(pred: str, refold: str, trb: str):
        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)

        trb = pickle.load(open(trb, 'rb'))
        c_r = np.unique(np.char.add(trb.chain_id[(trb.condition_token_mask) & (~trb.hetero)], np.array(trb.res_id[(trb.condition_token_mask) & (~trb.hetero)], dtype=str)))
        pred_structure_c_r = pred_structure[np.char.add(pred_structure.chain_id, np.array(pred_structure.res_id, dtype=str)).isin(c_r)]
        refold_structure_c_r = refold_structure[np.char.add(refold_structure.chain_id, np.array(refold_structure.res_id, dtype=str)).isin(c_r)]
        c_r_bb_mask = np.isin(refold_structure_c_r.atom_name, ["N", "CA", "C"])
        pred_coord_align, _ = struc.superimpose(refold_structure_c_r.coord, pred_structure_c_r.coord, c_r_bb_mask)
        atomic_motif_rmsd = struc.rmsd(refold_structure_c_r.coord, pred_coord_align)
        return atomic_motif_rmsd

    @staticmethod
    def compute_motif_rmsd(pred: str, refold: str, trb: str):
        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)
        
        trb = pickle.load(open(trb, 'rb'))
        motif = np.unique(np.char.add(trb.chain_id[trb.condition_token_mask], np.array(trb.res_id[trb.condition_token_mask], dtype=str)))
        pred_structure_motif_backbone = pred_structure[np.char.add(pred_structure.chain_id, np.array(pred_structure.res_id, dtype=str)).isin(motif) & (pred_structure.atom_name.isin(["N", "CA", "C"]))]
        refold_structure_motif_backbone = refold_structure[np.char.add(refold_structure.chain_id, np.array(refold_structure.res_id, dtype=str)).isin(motif) & (refold_structure.atom_name.isin(["N", "CA", "C"]))]
        pred_coord_align, _ = struc.superimpose(refold_structure_motif_backbone.coord, pred_structure_motif_backbone.coord)
        motif_rmsd = struc.rmsd(refold_structure_motif_backbone.coord, pred_coord_align)
        return motif_rmsd
    
    @staticmethod
    def compute_protein_align_nuc_rmsd(pred: str, refold: str, trb: str):
        pred_structure = RMSDCalculator._load_structure(pred)
        refold_structure = RMSDCalculator._load_structure(refold)

        trb = pickle.load(open(trb, 'rb'))
        cond_chain = str(trb.chain_id[trb.condition_token_mask][0]).strip()

        # AF3 can add missing side-chain atoms and OXT to the protein chain.
        # Align on shared protein CA residues instead of assuming full-atom
        # arrays are identical between the reference and refolded structures.
        pred_ca_map = RMSDCalculator._residue_atom_coord_map(
            pred_structure,
            atom_name="CA",
            chain_ids=[cond_chain],
            hetero=False,
        )
        refold_ca_map = RMSDCalculator._residue_atom_coord_map(
            refold_structure,
            atom_name="CA",
            chain_ids=[cond_chain],
            hetero=False,
        )
        refold_ca_coords, pred_ca_coords, shared_ca = RMSDCalculator._shared_coord_arrays(
            refold_ca_map,
            pred_ca_map,
        )
        if len(shared_ca) == 0:
            raise ValueError(
                f"No shared protein CA residues found on condition chain {cond_chain}: "
                f"pred={len(pred_ca_map)} ref={len(refold_ca_map)}"
            )

        _, transform = struc.superimpose(refold_ca_coords, pred_ca_coords)

        pred_c4_map = RMSDCalculator._residue_atom_coord_map(
            pred_structure,
            atom_name="C4'",
            chain_ids=None,
            hetero=False,
        )
        refold_c4_map = RMSDCalculator._residue_atom_coord_map(
            refold_structure,
            atom_name="C4'",
            chain_ids=None,
            hetero=False,
        )
        refold_c4_coords, pred_c4_coords, shared_c4 = RMSDCalculator._shared_coord_arrays(
            refold_c4_map,
            pred_c4_map,
        )
        if len(shared_c4) == 0:
            raise ValueError(
                f"No shared nucleic C4' residues found after aligning condition chain {cond_chain}: "
                f"pred={len(pred_c4_map)} ref={len(refold_c4_map)}"
            )

        pred_c4_coords_aligned = transform.apply(pred_c4_coords)
        align_nuc_rmsd = struc.rmsd(refold_c4_coords, pred_c4_coords_aligned)
        return align_nuc_rmsd
    
    # @staticmethod
    # def compute_nuc_align_ligand_rmsd(pred: str, refold: str, trb: str):
    #     if pred.endswith('.cif'):
    #         pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
    #     else:
    #         pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
    #     if refold.endswith('.cif'):
    #         refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
    #     else:
    #         refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)

    #     # cond_chain = 'A'
    #     trb = pickle.load(open(trb, 'rb'))
    #     cond_chain = trb.chain_id[trb.condition_token_mask][0]
    
    #     pred_structure_c1_and_cond_and_ligand = pred_structure[((pred_structure.chain_id == cond_chain) & (pred_structure.atom_name == "C1'"))| (pred_structure.hetero == True)]
    #     refold_structure_c1_and_cond_ligand = refold_structure[((refold_structure.chain_id == cond_chain) & (refold_structure.atom_name == "C1'")) | (refold_structure.hetero == True)]
    #     cond_mask = pred_structure_c1_and_cond_and_ligand.chain_id == cond_chain
    #     is_ligand_mask = pred_structure_c1_and_cond_and_ligand.hetero == True
    #     pred_coord_align, _ = struc.superimpose(pred_structure_c1_and_cond_and_ligand.coord, refold_structure_c1_and_cond_ligand.coord, cond_mask)

    #     align_nuc_rmsd = struc.rmsd(refold_structure_c1_and_cond_ligand.coord[is_ligand_mask], pred_coord_align[is_ligand_mask])
    #     return align_nuc_rmsd

    @staticmethod
    def compute_nuc_align_ligand_rmsd(pred: str, refold: str):
        if pred.endswith('.pdb'):
            gen_file = pdb.PDBFile.read(pred)
            gen_arr = pdb.get_structure(gen_file, model=1)
        else:
            gen_file = pdbx.CIFFile.read(pred)
            gen_arr = pdbx.get_structure(gen_file, model=1)
        
        refold_file = pdbx.CIFFile.read(refold)
        refold_arr = pdbx.get_structure(refold_file, model=1)

        is_ligand_mask = refold_arr.hetero == True

        chains_to_design = refold_arr.chain_id[is_ligand_mask][0]

        # breakpoint()

        refold_coord = np.concatenate(
            (
                refold_arr.coord[get_nuc_centre_atom_mask(refold_arr)], 
                refold_arr.coord[(refold_arr.chain_id == chains_to_design)]
            ), axis=0
        )

        gen_coord = np.concatenate(
            (
                gen_arr.coord[get_nuc_centre_atom_mask(gen_arr)], 
                gen_arr.coord[(gen_arr.chain_id == chains_to_design)]
            ), axis=0
        )

        lig_mask = np.concatenate(
            (
                np.zeros(get_nuc_centre_atom_mask(gen_arr).sum(), dtype=np.bool_),
                np.ones((gen_arr.chain_id == chains_to_design).sum(), dtype=np.bool_)
            ), axis=0
        )

        gen_coord_align, _ = struc.superimpose(refold_coord, gen_coord, atom_mask=~lig_mask)
        rmsd = struc.rmsd(refold_coord[lig_mask], gen_coord_align[lig_mask])

        return rmsd

NA_STD_RESIDUES_RES_NAME_TO_ONE = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "DA": "A",
    "DG": "G",
    "DC": "C",
    "DT": "T",    
}

def get_nuc_centre_atom_mask(atom_array):
    nuc_centre_atom_mask = (
        struc.filter_nucleotides(atom_array) & 
        np.array([atom.res_name in NA_STD_RESIDUES_RES_NAME_TO_ONE for atom in atom_array]) &
        (atom_array.atom_name == r"C1'")
    )
    return nuc_centre_atom_mask
