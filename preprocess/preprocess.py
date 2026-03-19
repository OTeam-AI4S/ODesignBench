import os
import shutil
import numpy as np
from pathlib import Path
from collections import Counter
from time import perf_counter
from typing import Any
import biotite.structure.io as io
from biotite.structure.io import pdbx, pdb
from biotite.structure import get_residue_starts

class Preprocess():

    def __init__(self, config):
        self.config = config

    def _make_result(
        self,
        stage: str,
        output_dir: str | None = None,
        success: bool = True,
        file_count: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "success": bool(success),
            "stage": stage,
            "outputs": {},
            "details": details or {},
        }
        if output_dir is not None:
            result["outputs"]["output_dir"] = str(output_dir)
            if file_count is None and os.path.isdir(output_dir):
                file_count = len([p for p in Path(output_dir).rglob("*") if p.is_file()])
        if file_count is not None:
            result["details"]["file_count"] = int(file_count)
        return result

    def run(self, action: str, **kwargs) -> dict[str, Any]:
        dispatch = {
            "format_output_pdb": self.format_output_pdb,
            "format_output_ligand": self.format_output_ligand,
            "format_output_cif": self.format_output_cif,
            "format_output_ligand_for_pbl_eval": self.format_output_ligand_for_protein_binding_ligand_evaluation,
            "format_output_ligand_for_protein_binding_ligand_evaluation": self.format_output_ligand_for_protein_binding_ligand_evaluation,
            "format_output_for_foldseek": self.format_output_for_foldseek,
            "rna_preprocess": self.rna_preprocess,
        }
        if action not in dispatch:
            known = ", ".join(sorted(dispatch.keys()))
            raise ValueError(f"Unknown preprocess action '{action}'. Supported: {known}")
        t0 = perf_counter()
        result = dispatch[action](**kwargs)
        elapsed = perf_counter() - t0
        if isinstance(result, dict):
            details = result.setdefault("details", {})
            details["elapsed_seconds"] = round(elapsed, 3)
        print(f"[timing] preprocess.{action}: {elapsed:.2f}s")
        return result


    def format_output_cif(self, input_dir: str, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)
        for case_dir in Path(input_dir).iterdir():
            for cif_path in case_dir.rglob("*.cif"):
                # Preserve original filename instead of renaming
                original_name = cif_path.name
                output_path = os.path.join(output_dir, original_name)
                shutil.copy(cif_path, output_path)
                print(f"Copied CIF file to {output_path}")
        return self._make_result(stage="preprocess.format_output_cif", output_dir=output_dir)

    def format_output_pdb(self, input_dir: str, output_dir: str, use_trb_for_b_factor: bool = True):
        """
        Collect PDB/CIF from design_dir: one subdir per target, each with .pdb or .cif design outputs.
        
        For AME (Atomic Motif Enzyme) tasks, if trb files are available, this function will:
        - Read motif residues from trb['con_hal_pdb_idx'] (these are residues to design, b_factor=0)
        - Set all other residues as fixed (b_factor=1.0)
        
        This matches the logic used in RFD2's AME module (parse_multiple_chains.py).
        
        Args:
            input_dir: Input directory containing PDB/CIF files (and optionally .trb files)
            output_dir: Output directory for formatted structures
            use_trb_for_b_factor: If True, try to read motif residues from .trb files to set b_factor correctly
        """
        os.makedirs(output_dir, exist_ok=True)
        input_path = Path(input_dir)

        # Case 1: Standard layout – one subdirectory per target (preserve original filenames)
        for case_dir in input_path.iterdir():
            if not case_dir.is_dir():
                continue
            for cif_path in case_dir.rglob("*.cif"):
                # Preserve original filename (change extension to .pdb)
                original_name = cif_path.stem + ".pdb"
                output_path = os.path.join(output_dir, original_name)
                atom_array = self._process_structure_with_b_factor(str(cif_path), use_trb_for_b_factor)
                atom_array = self._ensure_valid_b_factor(atom_array)
                io.save_structure(output_path, atom_array)
                print(f"Converted CIF to PDB and saved to {output_path}")
            for pdb_path in case_dir.rglob("*.pdb"):
                # Preserve original filename
                original_name = pdb_path.name
                output_path = os.path.join(output_dir, original_name)
                atom_array = self._process_structure_with_b_factor(str(pdb_path), use_trb_for_b_factor)
                atom_array = self._ensure_valid_b_factor(atom_array)
                io.save_structure(output_path, atom_array)
                print(f"Copied PDB to {output_path}")

        # Case 2: Flat layout – PDB/CIF files directly under input_dir or in a single subdirectory
        # This makes it easier to run AME on a small custom set of backbones
        # Check if there's only one subdirectory with files (common case: structures/ subdirectory)
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and len(list(input_path.glob("*.pdb"))) == 0 and len(list(input_path.glob("*.cif"))) == 0:
            # Single subdirectory with no files at root level - treat as flat layout
            # Use original filenames to preserve meaningful names for CSV matching
            single_subdir = subdirs[0]
            for cif_path in single_subdir.rglob("*.cif"):
                original_name = cif_path.stem
                output_path = os.path.join(output_dir, f"{original_name}.pdb")
                atom_array = self._process_structure_with_b_factor(str(cif_path), use_trb_for_b_factor)
                atom_array = self._ensure_valid_b_factor(atom_array)
                io.save_structure(output_path, atom_array)
                print(f"Converted CIF to PDB and saved to {output_path}")
            for pdb_path in single_subdir.rglob("*.pdb"):
                original_name = pdb_path.stem
                output_path = os.path.join(output_dir, f"{original_name}.pdb")
                atom_array = self._process_structure_with_b_factor(str(pdb_path), use_trb_for_b_factor)
                atom_array = self._ensure_valid_b_factor(atom_array)
                io.save_structure(output_path, atom_array)
                print(f"Copied PDB to {output_path}")
        else:
            # True flat layout – PDB/CIF files directly under input_dir
            flat_pdbs = list(input_path.glob("*.pdb"))
            flat_cifs = list(input_path.glob("*.cif"))
            # Use original filename (without extension) for flat layouts to preserve meaningful names
            # Fallback to folder name if needed

            for pdb_path in flat_pdbs:
                # Use original filename (without extension) as the output name
                original_name = pdb_path.stem
                output_path = os.path.join(output_dir, f"{original_name}.pdb")
                atom_array = self._process_structure_with_b_factor(str(pdb_path), use_trb_for_b_factor)
                atom_array = self._ensure_valid_b_factor(atom_array)
                io.save_structure(output_path, atom_array)
                print(f"Copied PDB to {output_path}")

            for cif_path in flat_cifs:
                # Use original filename (without extension) as the output name
                original_name = cif_path.stem
                output_path = os.path.join(output_dir, f"{original_name}.pdb")
                atom_array = self._process_structure_with_b_factor(str(cif_path), use_trb_for_b_factor)
                atom_array = self._ensure_valid_b_factor(atom_array)
                io.save_structure(output_path, atom_array)
                print(f"Converted CIF to PDB and saved to {output_path}")
        return self._make_result(
            stage="preprocess.format_output_pdb",
            output_dir=output_dir,
            details={"use_trb_for_b_factor": bool(use_trb_for_b_factor)},
        )
    
    def _process_structure_with_b_factor(self, struct_path: str, use_trb: bool = True):
        """
        Read structure and set b_factor based on motif residues from trb file (if available).
        
        Logic (matching RFD2's AME module):
        - Motif residues (from trb['con_hal_pdb_idx']) = need design (b_factor=0)
        - All other residues = fixed (b_factor=1.0)
        
        If no trb file is found or use_trb=False, preserves existing b_factor or defaults to 1.0.
        """
        atom_array = read_structure(struct_path)
        
        if not use_trb:
            # If b_factor doesn't exist or is all zeros, set default to 1.0 (all fixed)
            if not hasattr(atom_array, 'b_factor') or np.all(atom_array.b_factor == 0):
                atom_array.b_factor = np.ones_like(atom_array.chain_id, dtype=float)
            else:
                # Clip existing b_factor values to PDB format limits
                atom_array.b_factor = np.clip(atom_array.b_factor, 0.0, 999.99)
            return atom_array
        
        # Try to find corresponding trb file
        struct_path_obj = Path(struct_path)
        trb_path = struct_path_obj.with_suffix('.trb')
        
        # Also check in parent directory (common for RFD2 outputs)
        if not trb_path.exists():
            trb_path = struct_path_obj.parent / f"{struct_path_obj.stem}.trb"
        
        if trb_path.exists():
            try:
                trb = np.load(str(trb_path), allow_pickle=True)
                
                # Get motif residues from con_hal_pdb_idx (format: [(chain_id, res_id), ...])
                # Also check receptor_con_hal_pdb_idx if present (for receptor-ligand designs)
                motif_residues = set()
                
                if 'con_hal_pdb_idx' in trb:
                    for item in trb['con_hal_pdb_idx']:
                        # Handle different formats: (chain_id, res_id) or [chain_id, res_id]
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            chain_id, res_id = item[0], item[1]
                        else:
                            continue
                        
                        # Handle res_id format (could be int, tuple, etc.)
                        if isinstance(res_id, (tuple, list)):
                            res_id = res_id[1] if len(res_id) > 1 else res_id[0]
                        motif_residues.add((str(chain_id), int(res_id)))
                
                # Also include receptor motif residues if present
                if 'receptor_con_hal_pdb_idx' in trb:
                    for item in trb['receptor_con_hal_pdb_idx']:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            chain_id, res_id = item[0], item[1]
                            if isinstance(res_id, (tuple, list)):
                                res_id = res_id[1] if len(res_id) > 1 else res_id[0]
                            motif_residues.add((str(chain_id), int(res_id)))
                
                if motif_residues:
                    # Set b_factor: 0 for motif residues (design), 1.0 for others (fixed)
                    atom_array.b_factor = np.ones(len(atom_array), dtype=float)
                    
                    # Find atoms that belong to motif residues
                    for i, (chain_id, res_id) in enumerate(zip(atom_array.chain_id, atom_array.res_id)):
                        # Handle res_id format (could be int or tuple)
                        pdb_res_id = res_id[1] if isinstance(res_id, tuple) else res_id
                        if (str(chain_id), int(pdb_res_id)) in motif_residues:
                            atom_array.b_factor[i] = 0.0
                    
                    print(f"  Set b_factor from trb file: {len(motif_residues)} motif residues (b_factor=0), "
                          f"{np.sum(atom_array.b_factor == 1.0)} fixed residues (b_factor=1.0)")
                else:
                    print(f"  Warning: trb file found but no motif residues found in 'con_hal_pdb_idx' or 'receptor_con_hal_pdb_idx'. Using default b_factor=1.0")
                    atom_array.b_factor = np.ones(len(atom_array), dtype=float)
            except Exception as e:
                print(f"  Warning: Failed to read trb file {trb_path}: {e}. Using default b_factor=1.0")
                atom_array.b_factor = np.ones(len(atom_array), dtype=float)
        else:
            # No trb file found - preserve existing b_factor or default to 1.0
            if not hasattr(atom_array, 'b_factor') or np.all(atom_array.b_factor == 0):
                atom_array.b_factor = np.ones(len(atom_array), dtype=float)
                print(f"  No trb file found for {struct_path_obj.name}. Using default b_factor=1.0 (all fixed)")
            else:
                # Clip existing b_factor values to PDB format limits before preserving
                atom_array.b_factor = np.clip(atom_array.b_factor, 0.0, 999.99)
                print(f"  No trb file found for {struct_path_obj.name}. Preserving existing b_factor (clipped to 0-999.99)")
        
        # Ensure b_factor values are within PDB format limits (0-999.99)
        # Clip values to prevent biotite save errors (double-check all paths)
        if hasattr(atom_array, 'b_factor'):
            atom_array.b_factor = np.clip(atom_array.b_factor, 0.0, 999.99)
        
        return atom_array
    
    def _ensure_valid_b_factor(self, atom_array):
        """
        Ensure b_factor values are valid for PDB format before saving.
        This is a final safety check to prevent biotite save errors.
        """
        if not hasattr(atom_array, 'b_factor'):
            # If no b_factor, set default to 1.0
            atom_array.b_factor = np.ones(len(atom_array), dtype=np.float32)
        else:
            # Convert to float32 and clip to valid range
            atom_array.b_factor = np.clip(atom_array.b_factor.astype(np.float32), 0.0, 999.99)
            # Replace any NaN or Inf values
            atom_array.b_factor = np.nan_to_num(atom_array.b_factor, nan=1.0, posinf=999.99, neginf=0.0)
        return atom_array
                
    # need pdb
    def format_output_for_foldseek(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for cif_path in Path(input_dir).rglob("*.cif"):
            convert_cif_to_pdb_dir(cif_path=str(cif_path), struct_output_dir=output_dir)
        return self._make_result(stage="preprocess.format_output_for_foldseek", output_dir=output_dir)

    def format_output_ligand(self, input_dir: str, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        for case_dir in Path(input_dir).iterdir():
            for cif_path in case_dir.rglob("*.cif"):
                # Preserve original filename instead of renaming
                original_name = cif_path.name
                output_path = os.path.join(output_dir, original_name)
                # trb_path = os.path.join(os.path.dirname(cif_path), "traceback.pkl")
                # output_trb_path = os.path.join(output_dir, original_name.replace(".cif", ".pkl"))
                cif_file = pdbx.CIFFile.read(cif_path)
                atom_array = pdbx.get_structure(cif_file, model=1, extra_fields=['b_factor', 'occupancy'])
                atom_array.atom_name[atom_array.hetero] = np.char.add(['C']*sum(atom_array.hetero), np.array(range(sum(atom_array.hetero)), dtype=np.str_))
                io.save_structure(output_path, atom_array)
                # shutil.copy(trb_path, output_trb_path)
        return self._make_result(stage="preprocess.format_output_ligand", output_dir=output_dir)
    
    def format_output_ligand_for_protein_binding_ligand_evaluation(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for cif_path in Path(input_dir).rglob("*.cif"):
            output_path = os.path.join(output_dir, cif_path.name)  
            cif_file = pdbx.CIFFile.read(cif_path)
            block = cif_file.block
            atom_site = block.get("atom_site")
            atom_site["occupancy"] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
            atom_site['B_iso_or_equiv'] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
            cif_file.write(output_path)
        return self._make_result(
            stage="preprocess.format_output_ligand_for_pbl_eval",
            output_dir=output_dir,
        )
    
    def rna_preprocess(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        count = Counter()
        for case_dir in Path(input_dir).iterdir():
            case = case_dir.name
            for cif_path in case_dir.rglob('*.cif'):
                count[case] += 1
                cif_file = pdbx.CIFFile.read(cif_path)
                arr = pdbx.get_structure(cif_file, model=1)
                arr.atom_name[(arr.atom_name == 'N') & (arr.res_name == 'C')] = "N9" # concerned gRNAde
                pdb_file = pdb.PDBFile()
                pdb_file.set_structure(arr)
                pdb_file.write(os.path.join(output_dir, f'{case}_{count[case]}.pdb'))
                shutil.copy(os.path.join(os.path.dirname(cif_path), "traceback.pkl"), os.path.join(output_dir, f"{case}_{count[case]}.pkl"))
        return self._make_result(
            stage="preprocess.rna_preprocess",
            output_dir=output_dir,
            details={"num_cases": len(count), "num_structures": int(sum(count.values()))},
        )

    def make_ligandmpnn_input(self):

        pass
    
    def make_af3_input(self):

        pass
    
    def make_chai1_input(self):

        pass


import gemmi
import biotite.structure.io.pdbx as pdbx
def convert_cif_to_pdb(cif_path: str, struct_output_path: str) -> str:
    """Convert mmCIF file to PDB format with B_iso_or_equiv support"""
    pdb_path = struct_output_path
    
    if not os.path.exists(pdb_path):
        try:
            # First, check if B_iso_or_equiv exists in the CIF file
            cif_file = pdbx.CIFFile.read(cif_path)
            block = cif_file.block
            atom_site = block.get("atom_site")
            
            # Check if B_iso_or_equiv attribute exists, if not add it
            need_temp_file = False
            if "B_iso_or_equiv" not in atom_site:
                print(f"Adding B_iso_or_equiv to {cif_path}")
                atom_site["B_iso_or_equiv"] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
                need_temp_file = True
            else:
                print(f"B_iso_or_equiv already exists in {cif_path}")
            
            # Create temporary CIF file with B_iso_or_equiv if needed
            temp_cif_path = cif_path
            if need_temp_file:
                temp_cif_path = os.path.join(os.path.dirname(pdb_path), f"temp_{os.path.basename(cif_path)}")
                cif_file.write(temp_cif_path)
            
            # Use gemmi Python API to convert mmCIF to PDB
            structure = gemmi.read_structure(temp_cif_path)
            structure.write_pdb(pdb_path)
            
            # Clean up temporary file if created
            if temp_cif_path != cif_path and os.path.exists(temp_cif_path):
                os.remove(temp_cif_path)
                
            print(f"Successfully converted {cif_path} to {pdb_path}")
            
        except Exception as e:
            print(f"Error converting {cif_path} to PDB using gemmi API: {e}")
            # Fallback to command line if API fails
            print(f"Falling back to command line conversion: gemmi convert {cif_path} {pdb_path}")
            os.system(f"gemmi convert {cif_path} {pdb_path}")
    else:
        print(f"PDB file already exists: {pdb_path}")
    
    return pdb_path

def convert_cif_to_pdb_dir(cif_path: str, struct_output_dir: str) -> str:
    """Convert mmCIF file to PDB format with B_iso_or_equiv support"""
    filename = os.path.basename(os.path.splitext(cif_path)[0]) + '.pdb'
    pdb_path = os.path.join(struct_output_dir, filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(struct_output_dir, exist_ok=True)
    
    if not os.path.exists(pdb_path):
        try:
            # First, check if B_iso_or_equiv exists in the CIF file
            cif_file = pdbx.CIFFile.read(cif_path)
            block = cif_file.block
            atom_site = block.get("atom_site")
            
            # Check if B_iso_or_equiv attribute exists, if not add it
            need_temp_file = False
            if "B_iso_or_equiv" not in atom_site:
                print(f"Adding B_iso_or_equiv to {cif_path}")
                atom_site["B_iso_or_equiv"] = pdbx.CIFColumn(pdbx.CIFData(["1.0" for _ in range(len(atom_site['group_PDB']))]))
                need_temp_file = True
            else:
                print(f"B_iso_or_equiv already exists in {cif_path}")
            
            # Create temporary CIF file with B_iso_or_equiv if needed
            temp_cif_path = cif_path
            if need_temp_file:
                temp_cif_path = os.path.join(struct_output_dir, f"temp_{os.path.basename(cif_path)}")
                cif_file.write(temp_cif_path)
            
            # Use gemmi Python API to convert mmCIF to PDB
            structure = gemmi.read_structure(temp_cif_path)
            structure.write_pdb(pdb_path)
            
            # Clean up temporary file if created
            if temp_cif_path != cif_path and os.path.exists(temp_cif_path):
                os.remove(temp_cif_path)
                
            print(f"Successfully converted {cif_path} to {pdb_path}")
            
        except Exception as e:
            print(f"Error converting {cif_path} to PDB using gemmi API: {e}")
            # Fallback to command line if API fails
            print(f"Falling back to command line conversion: gemmi convert {cif_path} {pdb_path}")
            os.system(f"gemmi convert {cif_path} {pdb_path}")
    else:
        print(f"PDB file already exists: {pdb_path}")
    
    return pdb_path

def read_structure(fpath: str):
    """Read structure from a file (PDB or mmCIF)"""
    ext = os.path.splitext(fpath)[1].lower()
    if ext == '.pdb':
        return pdb.PDBFile.read(fpath).get_structure(model=1, extra_fields=["b_factor"])
    elif ext in ['.cif', '.mmcif']:
        cif_file = pdbx.CIFFile.read(fpath)
        return pdbx.get_structure(cif_file, model=1, extra_fields=["b_factor"])
    else:
        raise ValueError(f"Unsupported file extension: {ext}")