import pickle
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
        # Placeholder for RMSD calculation logic
        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)
        
        pred_bb_mask = (np.isin(pred_structure.atom_name, ["N", "CA", "C", "O"]) & (~pred_structure.hetero))
        refold_bb_mask = (np.isin(refold_structure.atom_name, ["N", "CA", "C", "O"]) & (~refold_structure.hetero))
        pred_coord_align, _ = struc.superimpose(refold_structure.coord[refold_bb_mask], pred_structure.coord[pred_bb_mask])
        bb_rmsd = struc.rmsd(refold_structure.coord[refold_bb_mask], pred_coord_align)
        return bb_rmsd
    
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
        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)

        def _ca_coord_map(arr) -> Dict[Tuple[str, int], np.ndarray]:
            mask = (arr.atom_name == "CA") & (~arr.hetero)
            out: Dict[Tuple[str, int], np.ndarray] = {}
            if np.sum(mask) == 0:
                return out
            for chain_id, res_id, coord in zip(arr.chain_id[mask], arr.res_id[mask], arr.coord[mask]):
                key = (str(chain_id), int(res_id))
                # If duplicates exist, keep the first.
                if key not in out:
                    out[key] = coord
            return out

        pred_map = _ca_coord_map(pred_structure)
        ref_map = _ca_coord_map(refold_structure)
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

        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)

        def _ca_coord_map(arr) -> Dict[Tuple[str, int], np.ndarray]:
            mask = (arr.atom_name == "CA") & (~arr.hetero) & (np.isin(arr.chain_id, list(chain_set)))
            out: Dict[Tuple[str, int], np.ndarray] = {}
            if np.sum(mask) == 0:
                return out
            for chain_id, res_id, coord in zip(arr.chain_id[mask], arr.res_id[mask], arr.coord[mask]):
                key = (str(chain_id), int(res_id))
                # If duplicates exist, keep the first.
                if key not in out:
                    out[key] = coord
            return out

        pred_map = _ca_coord_map(pred_structure)
        ref_map = _ca_coord_map(refold_structure)
        common = sorted(set(pred_map.keys()) & set(ref_map.keys()))
        if len(common) < 3:
            return float("nan")

        pred_coords = np.stack([pred_map[k] for k in common], axis=0)
        ref_coords = np.stack([ref_map[k] for k in common], axis=0)
        pred_align, _ = struc.superimpose(ref_coords, pred_coords)
        return float(struc.rmsd(ref_coords, pred_align))

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
        if pred.endswith('.cif'):
            pred_structure = pdbx.get_structure(pdbx.CIFFile.read(pred), model=1)
        else:
            pred_structure = pdb.get_structure(pdb.PDBFile.read(pred), model=1)
        
        if refold.endswith('.cif'):
            refold_structure = pdbx.get_structure(pdbx.CIFFile.read(refold), model=1)
        else:
            refold_structure = pdb.get_structure(pdb.PDBFile.read(refold), model=1)
        
        trb = pickle.load(open(trb, 'rb'))
        cond_chain = trb.chain_id[trb.condition_token_mask][0]
        pred_structure_c4_and_cond = pred_structure[(pred_structure.chain_id == cond_chain) | (pred_structure.atom_name == "C4'")]
        refold_structure_c4_and_cond = refold_structure[(refold_structure.chain_id == cond_chain) | (refold_structure.atom_name == "C4'")]
        cond_mask = pred_structure_c4_and_cond.chain_id == cond_chain
        c4_mask = pred_structure_c4_and_cond.atom_name == "C4'"
        pred_coord_align, _ = struc.superimpose(refold_structure_c4_and_cond.coord, pred_structure_c4_and_cond.coord, cond_mask)
        align_nuc_rmsd = struc.rmsd(refold_structure_c4_and_cond.coord[c4_mask], pred_coord_align[c4_mask])
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