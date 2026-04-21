import os
import json
import pickle
import subprocess
import numpy as np
from pathlib import Path
from time import perf_counter
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple, Callable
from biotite.structure.io import pdb, pdbx
import biotite.structure as struc
import biotite.structure.io as io
import sys

def _collect_unk_residues(pdb_path: Path) -> set[str]:
    """
    Collect residue IDs (e.g. "A123") whose residue name is UNK in a PDB file.
    """
    unk_residues: set[str] = set()
    with pdb_path.open("r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[17:20].strip() != "UNK":
                continue
            chain_id = line[21].strip()
            if not chain_id:
                continue
            try:
                res_id = int(line[22:26].strip())
            except ValueError:
                continue
            unk_residues.add(f"{chain_id}{res_id}")
    return unk_residues


def _remove_residues_from_fixed_list(fixed_residues: List[str], residues_to_remove: set[str]) -> List[str]:
    """Remove residue IDs from a fixed-residue list while preserving order."""
    if not residues_to_remove:
        return fixed_residues
    return [r for r in fixed_residues if r not in residues_to_remove]


def _replace_unk_with_ala(pdb_path: Path) -> bool:
    """
    Replace UNK residues with ALA in a PDB file.
    UNK (Unknown) residues cause issues with ProDy's select('protein') function,
    which doesn't recognize UNK as a valid amino acid.
    
    Returns True if replacements were made, False otherwise.
    """
    content = pdb_path.read_text()
    if 'UNK' not in content:
        return False
    
    # Replace UNK with ALA
    new_content = content.replace('UNK', 'ALA')
    
    # Write back
    pdb_path.write_text(new_content)
    return True


def _setup_distributed(local_rank, world_size):
    """Sets up the distributed environment for a worker process."""
    # Set master address and port (torchrun does this automatically)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Make sure this port is available

    dist.init_process_group(
        backend="nccl",      # NCCL for NVIDIA GPUs
        init_method="env://",
        rank=local_rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)

def _cleanup_distributed():
    """Cleans up the process group."""
    dist.destroy_process_group()

def _odesign_mpnn_worker(
    local_rank: int,
    world_size: int,
    config: Any,  # Your self.config object
    inverse_fold_root: str,
    samples_all: List[Dict[str, Any]]
):
    """
    This is the function that each individual GPU process will run.
    """
    from .InverseFold.utils.tools import reload_model, inference, _AA1_TO_AA3

    _setup_distributed(local_rank, world_size)
    device = f"cuda:{local_rank}"
    
    if local_rank == 0:
        print("Workers started. Loading models on each GPU...")

    # 1. Load the model (each process loads its own copy)
    model, dev = reload_model(
        data_name=config.inversefold.data_name, 
        model_name="ODesign", 
        device=device,
        model_dir=config.inversefold.model_dir
    )

    # 2. Get this process's unique data portion
    # We split the list based on rank: [rank::world_size]
    samples_for_this_rank = samples_all[local_rank::world_size]
    
    if local_rank == 0:
        print(f"Rank 0 processing {len(samples_for_this_rank)} samples (total {len(samples_all)}).")
    
    # 3. Run the inference loop (from your original function)
    for sample_dict in samples_for_this_rank:
        try:
            pred_seqs, scores, true_seq_masked, probs, metrics = inference(
                model=model, sample_input=sample_dict['smp'], model_name="ODesign",
                data_name=config.inversefold.data_name, topk=config.inversefold.topk,
                temp=config.inversefold.temp, use_beam=config.inversefold.use_beam,
                device=dev
            )

            # 4. Save results (from your original function)
            if config.inversefold.save_all:
                for i, seq in enumerate(pred_seqs):
                    # Use the path we saved in sample_dict
                    cif_src_path = sample_dict['cif_src_path'] 
                    # #qtfeng: Use original CIF filename (without extension) instead of smp['title'] to avoid filename conflicts
                    cif_basename = os.path.splitext(os.path.basename(cif_src_path))[0]
                    out_path = os.path.join(inverse_fold_root, f"{cif_basename}-{i+1}.cif")
                    
                    src_file = pdbx.CIFFile.read(cif_src_path)
                    src_atom_array = pdbx.get_structure(src_file, model=1)
                    
                    if sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'protein':
                        seq_1_to_3 = [_AA1_TO_AA3.get(rt, 'UNK') for rt in list(seq)]
                    elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'rna':
                        seq_1_to_3 = list(seq)
                    elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'dna':
                        seq_1_to_3 = [f"D{rt}" for rt in list(seq)]
                    else:
                        seq_1_to_3 = seq.split(' ')

                    save_sequence_replaced_array = src_atom_array.copy()
                    r_starts = struc.get_residue_starts(save_sequence_replaced_array, add_exclusive_stop=True)
                    redesign_id = 0
                    for r_s, r_e in zip(r_starts[:-1], r_starts[1:]):
                        if save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and config.inversefold.data_name != "ligand":
                            save_sequence_replaced_array.res_name[r_s:r_e] = seq_1_to_3[redesign_id]
                            redesign_id += 1
                        elif save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and config.inversefold.data_name == "ligand":
                            save_sequence_replaced_array.element[r_s: r_e] = seq_1_to_3
                            save_sequence_replaced_array.atom_name[r_s: r_e] = np.char.add(seq_1_to_3, np.array(range(r_e - r_s), dtype=np.str_))
                        else: continue

                    io.save_structure(out_path, save_sequence_replaced_array)

            # This update now happens locally in the process and will not be
            # visible in the main process. This is OK since the function anyway
            # did not return the list. The main result is the saved files.
            sample_dict["composition"][sample_dict["cond_c_id"]] = (sample_dict["composition"][sample_dict["cond_c_id"]][0], pred_seqs)
        
        except Exception as e:
            print(f"[Rank {local_rank}] Failed to process {sample_dict['smp']['title']}: {e}")

    _cleanup_distributed()
    print(f"Rank {local_rank} is done.")

class InverseFold:

    def __init__(self, config):

        self.config = config

    def _make_result(
        self,
        stage: str,
        output_dir: str | None = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": bool(success),
            "stage": stage,
            "outputs": {},
            "details": details or {},
        }
        if output_dir is not None:
            result["outputs"]["output_dir"] = str(output_dir)
        return result

    @staticmethod
    def _count_files(root_dir: str, patterns: List[str]) -> int:
        root = Path(root_dir)
        if not root.exists():
            return 0
        count = 0
        for pattern in patterns:
            count += len(list(root.rglob(pattern)))
        return count

    def run(self, action: str, **kwargs) -> Dict[str, Any]:
        dispatch = {
            "ligandmpnn_distributed": self.run_ligandmpnn_distributed,
            "ligandmpnn": self.run_ligandmpnn_distributed,
            "proteinmpnn_distributed": self.run_proteinmpnn_distributed,
            "proteinmpnn": self.run_proteinmpnn_distributed,
            "odesignmpnn": self.run_odesignmpnn,
            "odesign": self.run_odesignmpnn,
        }
        if action not in dispatch:
            known = ", ".join(sorted(dispatch.keys()))
            raise ValueError(f"Unknown inversefold action '{action}'. Supported: {known}")
        t0 = perf_counter()
        result = dispatch[action](**kwargs)
        elapsed = perf_counter() - t0
        if isinstance(result, dict):
            details = result.setdefault("details", {})
            details["elapsed_seconds"] = round(elapsed, 3)
        print(f"[timing] inversefold.{action}: {elapsed:.2f}s")
        return result

    def run_ligandmpnn_distributed(
        self, 
        input_dir: Path, 
        output_dir: str, 
        gpu_list: list, 
        origin_cwd: str, 
        cdr_info_csv: Optional[str] = None, 
        use_cdr_fix: bool = False,
        fixed_residues_calculator: Optional[Callable[[Path, Any], List[str]]] = None,
        cdr_df: Optional[Any] = None,
        ame_csv: Optional[str] = None,
        ame_csv_df: Optional[Any] = None,
        pbp_info_csv: Optional[str] = None,
        pbp_info_df: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run LigandMPNN in distributed mode.
        
        Args:
            input_dir: Directory containing PDB files
            output_dir: Output directory for results
            gpu_list: List of GPU IDs
            origin_cwd: Original working directory
            cdr_info_csv: Path to CDR info CSV (required if use_cdr_fix=True and cdr_df=None)
            use_cdr_fix: If True, use CDR information to fix scaffold residues (for antibodies)
            fixed_residues_calculator: Optional function to calculate fixed residues (pdb_path, cdr_row) -> List[str]
            cdr_df: Optional pre-loaded CDR DataFrame (if None, will load from cdr_info_csv)
            ame_csv: Path to AME CSV file (for Atomic Motif Enzyme tasks)
            ame_csv_df: Optional pre-loaded AME CSV DataFrame (if None, will load from ame_csv)
            pbp_info_csv: Path to PBP chain-role CSV file
            pbp_info_df: Optional pre-loaded PBP chain-role DataFrame (if None, will load from pbp_info_csv)
        """
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = Path(str(self.config.inversefold.checkpoint_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(origin_cwd) / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "ProteinMPNN checkpoint not found: "
                f"{checkpoint_path}. "
                "Set PROTEINMPNN_CHECKPOINT_PATH or inversefold.checkpoint_path, "
                "or download weights with: "
                "bash inversefold/LigandMPNN/get_model_params.sh inversefold/LigandMPNN/model_params"
            )
        my_env = os.environ.copy()
        ligandmpnn_multi_input = {}
        
        # Load CDR info if provided
        if use_cdr_fix and cdr_df is None:
            if cdr_info_csv is None:
                raise ValueError("cdr_info_csv is required when use_cdr_fix=True and cdr_df is not provided")
            from .cdr_utils import load_cdr_info_csv, match_pdb_to_cdr_info
            cdr_df = load_cdr_info_csv(cdr_info_csv)
            print(f"Loaded CDR info for {len(cdr_df)} antibodies")
        
        # Use provided calculator or default
        if fixed_residues_calculator is None and use_cdr_fix:
            from .cdr_utils import calculate_fixed_residues_for_antibody
            fixed_residues_calculator = calculate_fixed_residues_for_antibody
        
        # Load AME CSV if provided
        if ame_csv_df is None and ame_csv is not None:
            from .ame_csv_utils import load_ame_csv
            ame_csv_df = load_ame_csv(ame_csv)
            print(f"Loaded AME CSV info for {len(ame_csv_df)} structures")
        
        # Load PBP chain-role CSV if provided
        if pbp_info_df is None and pbp_info_csv is not None:
            from .pbp_csv_utils import load_pbp_info_csv
            pbp_info_df = load_pbp_info_csv(pbp_info_csv)
            print(f"Loaded PBP chain-role CSV info for {len(pbp_info_df)} structures")

        converted_dir = Path(output_dir) / "converted_pdb"
        converted_dir.mkdir(parents=True, exist_ok=True)

        def _ensure_pdb(struct_path: Path) -> Path | None:
            """Convert CIF to PDB if needed; return PDB path or None on failure."""
            if struct_path.suffix.lower() == ".cif":
                rel = struct_path.relative_to(input_dir) if input_dir in struct_path.parents else Path(struct_path.name)
                pdb_dest = converted_dir / rel.parent / f"{struct_path.stem}.pdb"
                pdb_dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cif_file = pdbx.CIFFile.read(str(struct_path))
                    atom_array = pdbx.get_structure(cif_file, model=1)
                    pdb_file = pdb.PDBFile()
                    pdb_file.set_structure(atom_array)
                    pdb_file.write(str(pdb_dest))
                    # Replace UNK with ALA for compatibility with ProDy's protein selection
                    _replace_unk_with_ala(pdb_dest)
                    return pdb_dest
                except Exception as e:
                    print(f"Warning: Failed to convert CIF {struct_path.name} to PDB: {e}, skipping")
                    return None
            # Replace UNK with ALA for PDB files as well
            _replace_unk_with_ala(struct_path)
            return struct_path

        struct_paths = list(input_dir.rglob("*.pdb")) + list(input_dir.rglob("*.cif"))
        for struct_path in struct_paths:
            pdb_path = _ensure_pdb(struct_path)
            if pdb_path is None:
                continue
            pdb_key = str(pdb_path.resolve())  # Use actual path for LigandMPNN to open
            
            # Priority: PBP CSV > AME CSV > CDR > b_factor
            if pbp_info_df is not None:
                from .pbp_csv_utils import match_pdb_to_pbp_info, calculate_fixed_residues_from_chain_roles
                pbp_info = match_pdb_to_pbp_info(struct_path, pbp_info_df)
                if pbp_info is not None:
                    try:
                        fixed_residues = calculate_fixed_residues_from_chain_roles(
                            pdb_path,
                            design_chain=pbp_info["design_chain"],
                            target_chain=pbp_info["target_chain"],
                        )
                        fixed_residues_str = " ".join(fixed_residues)
                        ligandmpnn_multi_input[pdb_key] = fixed_residues_str
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to calculate PBP CSV-based fixed residues for {struct_path.name}: {e}")
                        print("  Falling back to b_factor-based method")
                else:
                    print(f"Warning: No PBP CSV info found for {struct_path.name}, using b_factor method")

            if ame_csv_df is not None:
                from .ame_csv_utils import match_pdb_to_csv_info, calculate_fixed_residues_from_motif
                csv_info = match_pdb_to_csv_info(struct_path, ame_csv_df)
                if csv_info is not None:
                    try:
                        # Calculate fixed residues: all protein residues except motif residues
                        fixed_residues = calculate_fixed_residues_from_motif(struct_path, csv_info['motif_residues'])
                        fixed_residues_str = " ".join(fixed_residues)
                        ligandmpnn_multi_input[pdb_key] = fixed_residues_str
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to calculate AME motif-based fixed residues for {struct_path.name}: {e}")
                        print(f"  Falling back to b_factor-based method")
                else:
                    print(f"Warning: No AME CSV info found for {struct_path.name}, using b_factor method")
            
            if fixed_residues_calculator is not None and not use_cdr_fix:
                try:
                    fixed_residues = fixed_residues_calculator(struct_path, None)
                    fixed_residues_str = " ".join(fixed_residues)
                    ligandmpnn_multi_input[pdb_key] = fixed_residues_str
                    continue
                except Exception as e:
                    print(f"Warning: Failed custom fixed-residue calculation for {struct_path.name}: {e}")
                    print("  Falling back to b_factor-based method")

            if use_cdr_fix and cdr_df is not None and fixed_residues_calculator is not None:
                from .cdr_utils import match_pdb_to_cdr_info
                cdr_row = match_pdb_to_cdr_info(struct_path, cdr_df)  # match on original path (cdr id from filename)
                if cdr_row is not None:
                    try:
                        fixed_residues = fixed_residues_calculator(struct_path, cdr_row)
                        fixed_residues_str = " ".join(fixed_residues)
                        ligandmpnn_multi_input[pdb_key] = fixed_residues_str
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to calculate CDR-based fixed residues for {struct_path.name}: {e}")
                        print(f"  Falling back to b_factor-based method")
                else:
                    print(f"Warning: No CDR info found for {struct_path.name}, using b_factor method")
            
            # Fallback to b_factor method
            atom_array = pdb.PDBFile.read(str(pdb_path)).get_structure(model=1, extra_fields=['b_factor'])
            # Only include protein residues (exclude HETATM/ligand chains)
            # Filter to protein atoms only (hetero=False means ATOM, hetero=True means HETATM)
            protein_atoms = atom_array[~atom_array.hetero]
            fixed_atom_array = protein_atoms[protein_atoms.b_factor != 0.0]
            fixed_residues = np.unique(np.char.add(fixed_atom_array.chain_id, np.array(fixed_atom_array.res_id, dtype=str)))
            ligandmpnn_multi_input[pdb_key] = " ".join(fixed_residues.tolist())
        
        json.dump(ligandmpnn_multi_input, open(os.path.join(output_dir, "ligandmpnn_input.json"), 'w'), indent=4)

        if pbp_info_df is not None:
            print(f"Prepared fixed residues for {len(ligandmpnn_multi_input)} structures (PBP CSV-based)")
        elif ame_csv_df is not None:
            print(f"Prepared fixed residues for {len(ligandmpnn_multi_input)} structures (AME motif-based)")
        elif fixed_residues_calculator is not None and not use_cdr_fix:
            print(f"Prepared fixed residues for {len(ligandmpnn_multi_input)} structures (custom calculator)")
        elif use_cdr_fix and cdr_df is not None:
            print(f"Prepared fixed residues for {len(ligandmpnn_multi_input)} structures (CDR-based)")
        num_gpus = len(gpu_list)
        all_items_list = list(ligandmpnn_multi_input.items())
        data_batches = np.array_split(all_items_list, num_gpus)
        print(f"Split {len(all_items_list)} structures (from dict) into {num_gpus} batches for parallel processing.")

        def _run_batch_on_gpu(gpu_id: str, batch_item_list: list, batch_index: int):

            if len(batch_item_list) == 0:
                print(f"GPU {gpu_id} (Batch {batch_index}) has no work. Skipping.")
                return f"GPU {gpu_id} skipped."

            worker_output_dir = output_dir
            os.makedirs(worker_output_dir, exist_ok=True)
            
            worker_json_path = os.path.join(worker_output_dir, f"ligandmpnn_input_{gpu_id}.json")
            worker_data_dict = dict(batch_item_list) # <-- Reconstruct dictionary from (key, value) tuple list
            
            with open(worker_json_path, 'w') as f:
                json.dump(worker_data_dict, f) # <-- Write this new, smaller dictionary
            
            print(f"Starting batch {batch_index} on GPU {gpu_id} ({len(worker_data_dict)} structures)...")
            
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            py = sys.executable or "python3"
            cmd = (
                # f'conda run -n ligandmpnn python {self.config.inversefold.exec} '
                f'{py} {self.config.inversefold.exec} '
                f'--model_type {self.config.inversefold.model_type} '
                f'--checkpoint_{self.config.inversefold.model_type} {self.config.inversefold.checkpoint_path} '
                f'--pdb_path_multi {worker_json_path} '  # <-- [Modified] Use worker's JSON
                f'--out_folder {worker_output_dir} '     # <-- [Modified] Use worker's output directory
                f'--batch_size {self.config.inversefold.batch_size} '
                f'--number_of_batches {self.config.inversefold.number_of_batches} '
                # f'--pack_side_chains 1 '
                f'--pack_side_chains 0 '
                f'--number_of_packs_per_design 4 ' 
                f'--pack_with_ligand_context 0 ' 
                # f'--temperature {self.config.inversefold.temperature} '
                f'--seed {self.config.inversefold.seed} '
                f'--fixed_residues_multi "{worker_json_path}" ' # <-- [Modified] Use worker's JSON
                f'--parse_atoms_with_zero_occupancy {self.config.inversefold.parse_atoms_with_zero_occupancy}'
            )
            
            # 10. Run subprocess
            try:
                subprocess.run(cmd, shell=True, check=True, env=my_env, cwd=origin_cwd)
                return f"Batch {batch_index} (GPU {gpu_id}) finished successfully."
            except subprocess.CalledProcessError as e:
                print(f"Error processing batch {batch_index} on GPU {gpu_id}: {e}")
                raise e
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i in range(num_gpus):
                gpu_id = gpu_list[i]
                batch_data_items = data_batches[i].tolist() 
                
                futures.append(
                    executor.submit(
                        _run_batch_on_gpu, 
                        gpu_id, 
                        batch_data_items, 
                        i
                    )
                )

            print("Waiting for all LigandMPNN batches to complete...")
            for future in futures:
                try:
                    result_message = future.result() 
                    print(result_message)
                except Exception as e:
                    print(f"A LigandMPNN worker failed: {e}")

        print("All LigandMPNN parallel processing is complete.")
        backbone_count = self._count_files(os.path.join(output_dir, "backbones"), ["*.pdb", "*.cif"])
        seq_count = self._count_files(os.path.join(output_dir, "seqs"), ["*.fa", "*.fasta"])
        return self._make_result(
            stage="inversefold.run_ligandmpnn_distributed",
            output_dir=output_dir,
            details={
                "num_gpus": num_gpus,
                "num_inputs": len(all_items_list),
                "num_backbones": backbone_count,
                "num_seq_files": seq_count,
            },
        )
    
    def run_proteinmpnn_distributed(
        self,
        input_dir: Path,
        output_dir: str,
        gpu_list: list,
        origin_cwd: str,
        pbp_info_csv: Optional[str] = None,
        pbp_info_df: Optional[Any] = None,
        scaffold_info_csv: Optional[str] = None,
        scaffold_info_df: Optional[Any] = None,
        motif_info_csv: Optional[str] = None,
        motif_info_df: Optional[Any] = None,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = Path(str(self.config.inversefold.checkpoint_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(origin_cwd) / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "ProteinMPNN checkpoint not found: "
                f"{checkpoint_path}. "
                "Set PROTEINMPNN_CHECKPOINT_PATH or inversefold.checkpoint_path, "
                "or download weights with: "
                "bash inversefold/LigandMPNN/get_model_params.sh inversefold/LigandMPNN/model_params"
            )
        ligandmpnn_multi_input = {}
        
        if pbp_info_df is None and pbp_info_csv is not None:
            from .pbp_csv_utils import load_pbp_info_csv
            pbp_info_df = load_pbp_info_csv(pbp_info_csv)
            print(f"Loaded PBP chain-role CSV info for {len(pbp_info_df)} structures")
        if scaffold_info_df is None and scaffold_info_csv is not None:
            from .motif_scaffolding_utils import load_scaffold_info_csv
            scaffold_info_df = load_scaffold_info_csv(scaffold_info_csv)
            print(f"Loaded motif scaffold info for {len(scaffold_info_df)} structures")
        if motif_info_df is None and motif_info_csv is not None:
            from .motif_scaffolding_utils import load_motif_info_csv
            motif_info_df = load_motif_info_csv(motif_info_csv)
            print(f"Loaded motif_info for {len(motif_info_df)} structures")
        elif motif_info_df is None and scaffold_info_csv is not None:
            from .motif_scaffolding_utils import build_motif_info_rows
            motif_name = input_dir.name
            motif_pdbs_dir = None
            if hasattr(self.config, "motif_scaffolding") and hasattr(self.config.motif_scaffolding, "motif_pdbs_dir"):
                motif_pdbs_dir = Path(str(self.config.motif_scaffolding.motif_pdbs_dir))
            if motif_pdbs_dir is None:
                motif_pdbs_dir = Path(__file__).resolve().parent.parent / "assets" / "motif_scaffolding"
            motif_pdb_path = motif_pdbs_dir / f"{motif_name}.pdb"
            if motif_pdb_path.exists():
                motif_info_df = build_motif_info_rows(scaffold_info_csv, motif_name, motif_pdb_path)
                print(f"Built motif_info in memory for {len(motif_info_df)} structures")

        # Replace UNK with ALA before processing to ensure compatibility with ProDy
        # UNK residues cause ProDy's select('protein') to miss these positions
        unk_replaced_count = 0
        unk_residues_by_pdb: Dict[str, set[str]] = {}
        for pdb_path in input_dir.rglob("*.pdb"):
            unk_residues = _collect_unk_residues(pdb_path)
            if unk_residues:
                unk_residues_by_pdb[str(pdb_path)] = unk_residues
            # Replace UNK residues in-place for compatibility with downstream tools
            if _replace_unk_with_ala(pdb_path):
                unk_replaced_count += 1
                print(f"  Replaced UNK with ALA in {pdb_path.name}")
        if unk_replaced_count > 0:
            print(f"Total UNK->ALA replacements: {unk_replaced_count} files")

        for pdb_path in input_dir.rglob("*.pdb"):
            if pbp_info_df is not None:
                from .pbp_csv_utils import match_pdb_to_pbp_info, calculate_fixed_residues_from_chain_roles
                pbp_info = match_pdb_to_pbp_info(pdb_path, pbp_info_df)
                if pbp_info is None:
                    raise ValueError(
                        f"No PBP CSV info found for '{pdb_path.name}'. "
                        "Ensure design_name in pbp_info_csv matches each formatted design filename."
                    )
                fixed_residues = calculate_fixed_residues_from_chain_roles(
                    pdb_path,
                    design_chain=pbp_info["design_chain"],
                    target_chain=pbp_info["target_chain"],
                )
                fixed_residues = _remove_residues_from_fixed_list(
                    fixed_residues,
                    unk_residues_by_pdb.get(str(pdb_path), set()),
                )
                ligandmpnn_multi_input[str(pdb_path)] = " ".join(fixed_residues)
            elif scaffold_info_df is not None:
                from .motif_scaffolding_utils import (
                    calculate_redesigned_residues_from_motif_info,
                    match_pdb_to_scaffold_info,
                    match_pdb_to_motif_info,
                    calculate_fixed_residues_from_motif_placements,
                )
                scaffold_row = match_pdb_to_scaffold_info(pdb_path, scaffold_info_df)
                motif_row = match_pdb_to_motif_info(pdb_path, motif_info_df) if motif_info_df is not None else None
                if scaffold_row is None:
                    print(
                        f"Warning: No scaffold_info row found for '{pdb_path.name}', "
                        "falling back to b_factor method"
                    )
                else:
                    try:
                        fixed_residues = calculate_fixed_residues_from_motif_placements(
                            pdb_path,
                            scaffold_row
                        )
                        if motif_row is not None:
                            redesign_residues = set(
                                calculate_redesigned_residues_from_motif_info(pdb_path, motif_row)
                            )
                            fixed_residues = _remove_residues_from_fixed_list(
                                fixed_residues,
                                redesign_residues,
                            )
                        fixed_residues = _remove_residues_from_fixed_list(
                            fixed_residues,
                            unk_residues_by_pdb.get(str(pdb_path), set()),
                        )
                        ligandmpnn_multi_input[str(pdb_path)] = " ".join(fixed_residues)
                        continue
                    except Exception as e:
                        print(
                            f"Warning: Failed scaffold_info-based fixed residues for "
                            f"{pdb_path.name}: {e}"
                        )
                        print("  Falling back to b_factor-based method")

            if str(pdb_path) not in ligandmpnn_multi_input:
                atom_array = pdb.PDBFile.read(pdb_path).get_structure(model=1, extra_fields=['b_factor'])
                fixed_atom_array = atom_array[atom_array.b_factor != 0.0]
                fixed_residues = np.unique(np.char.add(fixed_atom_array.chain_id, np.array(fixed_atom_array.res_id, dtype=str))).tolist()
                fixed_residues = _remove_residues_from_fixed_list(
                    fixed_residues,
                    unk_residues_by_pdb.get(str(pdb_path), set()),
                )
                ligandmpnn_multi_input[str(pdb_path)] = " ".join(fixed_residues)
        json.dump(ligandmpnn_multi_input, open(os.path.join(output_dir, "ligandmpnn_input.json"), 'w'), indent=4)

        num_gpus = len(gpu_list)
        all_items_list = list(ligandmpnn_multi_input.items())
        data_batches = np.array_split(all_items_list, num_gpus)
        print(f"Split {len(all_items_list)} structures (from dict) into {num_gpus} batches for parallel processing.")

        def _run_batch_on_gpu(gpu_id: str, batch_item_list: list, batch_index: int):

            if len(batch_item_list) == 0:
                print(f"GPU {gpu_id} (Batch {batch_index}) has no work. Skipping.")
                return f"GPU {gpu_id} skipped."

            worker_output_dir = output_dir
            os.makedirs(worker_output_dir, exist_ok=True)
            
            worker_json_path = os.path.join(worker_output_dir, f"ligandmpnn_input_{gpu_id}.json")
            worker_data_dict = dict(batch_item_list) # <-- Reconstruct dictionary from (key, value) tuple list
            
            with open(worker_json_path, 'w') as f:
                json.dump(worker_data_dict, f) # <-- Write this new, smaller dictionary
            
            print(f"Starting batch {batch_index} on GPU {gpu_id} ({len(worker_data_dict)} structures)...")
            
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            py = sys.executable or "python3"
            cmd = (
                f'{py} {self.config.inversefold.exec} '
                f'--model_type {self.config.inversefold.model_type} '
                f'--checkpoint_{self.config.inversefold.model_type} {self.config.inversefold.checkpoint_path} '
                f'--pdb_path_multi {worker_json_path} '  # <-- [Modified] Use worker's JSON
                f'--out_folder {worker_output_dir} '     # <-- [Modified] Use worker's output directory
                f'--batch_size {self.config.inversefold.batch_size} '
                f'--number_of_batches {self.config.inversefold.number_of_batches} '
                f'--temperature {self.config.inversefold.temperature} '
                f'--seed {self.config.inversefold.seed} '
                f'--fixed_residues_multi "{worker_json_path}"' # <-- [Modified] Use worker's JSON
                f' --parse_atoms_with_zero_occupancy {self.config.inversefold.parse_atoms_with_zero_occupancy}'
            )
            
            # 10. Run subprocess
            try:
                subprocess.run(cmd, shell=True, check=True, env=my_env, cwd=origin_cwd)
                return f"Batch {batch_index} (GPU {gpu_id}) finished successfully."
            except subprocess.CalledProcessError as e:
                print(f"Error processing batch {batch_index} on GPU {gpu_id}: {e}")
                raise e
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i in range(num_gpus):
                gpu_id = gpu_list[i]
                batch_data_items = data_batches[i].tolist() 
                
                futures.append(
                    executor.submit(
                        _run_batch_on_gpu, 
                        gpu_id, 
                        batch_data_items, 
                        i
                    )
                )

            print("Waiting for all LigandMPNN batches to complete...")
            for future in futures:
                try:
                    result_message = future.result() 
                    print(result_message)
                except Exception as e:
                    print(f"A LigandMPNN worker failed: {e}")

        print("All LigandMPNN parallel processing is complete.")
        backbone_count = self._count_files(os.path.join(output_dir, "backbones"), ["*.pdb", "*.cif"])
        seq_count = self._count_files(os.path.join(output_dir, "seqs"), ["*.fa", "*.fasta"])
        return self._make_result(
            stage="inversefold.run_proteinmpnn_distributed",
            output_dir=output_dir,
            details={
                "num_gpus": num_gpus,
                "num_inputs": len(all_items_list),
                "num_backbones": backbone_count,
                "num_seq_files": seq_count,
            },
        )
    
    def run_odesignmpnn(self, input_root: Path, inverse_fold_root: str) -> Dict[str, Any]:
        """
        Starts ODesignMPNN inference on multiple GPUs.
        The 'device' parameter is ignored and is now controlled by 'local_rank'.
        """
        from .InverseFold.utils.tools import parse_cif_to_samples
        from .InverseFold.utils.inference_utils import extract_ligand_samples_from_cif

        os.makedirs(inverse_fold_root, exist_ok=True)
        samples_all: List[Dict[str, Any]] = []

        # --- STEP 1: Data preparation (runs on main process) ---
        print("Preparing samples...")
        for cif_path in input_root.rglob("*.cif"):
            # #qtfeng: Read B_iso_or_equiv from CIF file instead of pkl file
            # #qtfeng: If B_iso_or_equiv is 0, the chain needs design (mpnn)
            # #qtfeng: If B_iso_or_equiv is 1, the chain does not need mpnn
            try:
                cif_file = pdbx.CIFFile.read(cif_path)
                atom_array = pdbx.get_structure(cif_file, model=1, extra_fields=['b_factor'])
            except Exception as e:
                print(f"Warning: Failed to read CIF file {cif_path}: {e}, skipping")
                continue
            
            # #qtfeng: Get design chains based on B_iso_or_equiv (0 means need design)
            design_chains = []
            for chain_id in np.unique(atom_array.chain_id):
                chain_atoms = atom_array[atom_array.chain_id == chain_id]
                # #qtfeng: Check if all atoms in this chain have B_iso_or_equiv == 0 (need design)
                if np.all(chain_atoms.b_factor == 0.0):
                    design_chains.append(chain_id)
            
            # #qtfeng: Skip if no chains need design (all chains have B_iso_or_equiv == 1)
            if len(design_chains) == 0:
                print(f"Info: No chains need design in {cif_path} (all B_iso_or_equiv == 1), skipping")
                continue
            
            assert len(design_chains) == 1, f"Only one design chain is allowed, but got {len(design_chains)} in {cif_path}"
            
            if self.config.inversefold.data_name != 'ligand':
                smp = parse_cif_to_samples(cif_path=cif_path, data_name=self.config.inversefold.data_name, only_chain=None)
            else:
                smp = extract_ligand_samples_from_cif(cif_path=cif_path, only_chain=None, skip_waters=True)
            
            # breakpoint()

            sample_dict = {"smp": smp[0], "composition": {}}
            
            # *** IMPORTANT: Add the source path so the worker can find the CIF file ***
            sample_dict["cif_src_path"] = str(cif_path)

            # #qtfeng: Process chains based on B_iso_or_equiv from CIF file
            # #qtfeng: Get default mol_type from config
            default_mol_type = self.config.inversefold.data_name if self.config.inversefold.data_name != 'ligand' else 'ligand'
            for c_id in np.unique(atom_array.chain_id):
                c_array = atom_array[atom_array.chain_id == c_id]
                # #qtfeng: Get mol_type for this chain
                if hasattr(c_array, 'mol_type') and len(c_array.mol_type) > 0:
                    mol_type = str(c_array.mol_type[0])
                else:
                    mol_type = default_mol_type
                
                # #qtfeng: If all atoms have B_iso_or_equiv == 1, this chain does not need design
                if np.all(c_array.b_factor == 1.0):
                    # #qtfeng: This is a fixed chain, get its sequence
                    if mol_type == 'ligand':
                        # #qtfeng: For ligand, use res_name
                        sample_dict['composition'][str(c_id)] = (mol_type, c_array.res_name[0] if len(c_array.res_name) > 0 else 'UNK')
                    else:
                        sample_dict['composition'][str(c_id)] = (mol_type, str(struc.to_sequence(c_array)[0][0]))
                else:
                    # #qtfeng: This chain needs design (B_iso_or_equiv == 0)
                    sample_dict["cond_c_id"] = str(c_id)
                    sample_dict['composition'][str(c_id)] = (mol_type, "REDESIGN_FLAG")
            samples_all.append(sample_dict)

        if not samples_all:
            print("Found no CIF files to process. Exiting.")
            return self._make_result(
                stage="inversefold.run_odesignmpnn",
                output_dir=str(inverse_fold_root),
                success=False,
                details={"message": "No CIF files to process"},
            )

        print(f"Found {len(samples_all)} samples. Starting multi-GPU inference...")

        # --- STEP 2: Start Multi-GPU Workers ---
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("ERROR: No GPUs found. Exiting.")
            return self._make_result(
                stage="inversefold.run_odesignmpnn",
                output_dir=str(inverse_fold_root),
                success=False,
                details={"message": "No GPUs found"},
            )

        # Arguments to pass to each worker process
        # (local_rank is added automatically by mp.spawn)
        args_to_pass = (
            world_size,
            self.config,  # Pass the configuration object
            inverse_fold_root,
            samples_all,  # Pass the complete list
        )
        
        # breakpoint()

        mp.spawn(
            _odesign_mpnn_worker,  # Function to run
            args=args_to_pass,         # Arguments
            nprocs=world_size,         # Number of processes (one per GPU)
            join=True                  # Wait until all are done
        )
        
        print("Multi-GPU inference completed.")
        output_cif_count = self._count_files(inverse_fold_root, ["*.cif"])
        return self._make_result(
            stage="inversefold.run_odesignmpnn",
            output_dir=str(inverse_fold_root),
            details={"num_samples": len(samples_all), "world_size": world_size, "num_outputs": output_cif_count},
        )

    # def run_gRNAde(self, input_dir: Path, g_id: str):

    #     model = gRNAde(gpu_id=g_id)
    #     for pdb_path in Path(input_dir).rglob('*.pdb'):
    #         def resnames_to_seq_fallback(resnames, monomer):
    #             if monomer.upper() == 'P':
    #                 return ''.join(AA3_TO_1.get(str(r).upper(), 'X') for r in resnames)
    #             else:
    #                 return ''.join(NA3_TO_1.get(str(r).upper(), 'N') for r in resnames)
    #         sequence = []
    #         pdbfile = pdb.PDBFile.read(pdb_path)
    #         atom_array = pdb.get_structure(pdbfile, model=1)
    #         trb_path = pdb_path.parent / f"{pdb_path.stem}.trb"
    #         trb = pickle.load(open(trb_path, 'rb'))
    #         unique_chains = np.unique(atom_array.chain_id)
    #         for c_id in unique_chains:
    #             c_array = atom_array[atom_array.chain_id == c_id]
    #             c_trb = trb[trb.chain_id == c_id]
    #         sequences, samples, perplexity, recovery = model.design_from_pdb_file(pdb_path, n_samples=self.config.inversefold.n_samples, partial_seq=)
    #         sequences, samples, perplexity, recovery = model.design_from_pdb_file(pdb_path, n_samples=self.config.inversefold.n_samples, partial_seq=)
    #         for sample_id, seqrecord in enumerate(sequences):
    #             seq = [str(seqrecord.seq)] + sequence
    #             all_pred_seqs.append(':'.join(seq))
    #             all_names.append(f"{pdb_path.stem}_{sample_id+1}")

    
    # def run_odesignmpnn(self, input_root: Path, inverse_fold_root: str, device: str):
    #     """
    #     we only support one condition chain in the cif file for now
    #     """
    #     os.makedirs(inverse_fold_root, exist_ok=True)
    #     samples_all: List[Dict[str, Any]] = []

    #     for cif_path in input_root.rglob("*.cif"):
    #         trb_path = cif_path.parent / f"{cif_path.stem}.pkl"
    #         trb = pickle.load(open(trb_path, 'rb'))
    #         condition_chains = np.unique(trb.chain_id[trb.condition_token_mask])
    #         assert len(condition_chains) == 1, f"Only one condition chain is allowed, but got {condition_chains} in {cif_path}"
            
    #         if self.config.inversefold.data_name != 'ligand':
    #             smp = parse_cif_to_samples(cif_path=cif_path, data_name=self.config.inversefold.data_name, only_chain=None)
    #         else:
    #             smp = extract_ligand_samples_from_cif(cif_path=cif_path, only_chain=None, skip_waters=True)
    #         sample_dict = {"smp": smp[0], "composition": {}}
    #         for c_id in np.unique(trb.chain_id):
    #             c_array = trb[trb.chain_id == c_id]
    #             if all(c_array.condition_token_mask):
    #                 if c_array.is_ligand.all():
    #                     sample_dict['composition'][str(c_id)] = (str(c_array.mol_type[0]), c_array.res_name[0]) #(mol_type, ccdCode)
    #                 else:
    #                     sample_dict['composition'][str(c_id)] = (str(c_array.mol_type[0]), str(struc.to_sequence(c_array)[0][0]))
    #             else:
    #                 sample_dict["cond_c_id"] = str(c_id)
    #                 sample_dict['composition'][str(c_id)] = (str(c_array.mol_type[0]), "REDESIGN_FLAG")
    #         samples_all.append(sample_dict)

    #     model, dev = reload_model(data_name=self.config.inversefold.data_name, model_name="ODesign", device=device)
    #     for sample_dict in samples_all:
    #         pred_seqs, scores, true_seq_masked, probs, metrics = inference(
    #             model=model, sample_input=sample_dict['smp'], model_name="ODesign",
    #             data_name=self.config.inversefold.data_name, topk=self.config.inversefold.topk,
    #             temp=self.config.inversefold.temp, use_beam=self.config.inversefold.use_beam,
    #             device=dev
    #         )
    #         if self.config.inversefold.save_all:
    #             for i, seq in enumerate(pred_seqs):
    #                 cif_src_path=input_root / f"{sample_dict['smp']['title']}.cif"
    #                 out_path=os.path.join(inverse_fold_root, f"{sample_dict['smp']['title']}-{i+1}.cif")
    #                 src_file = pdbx.CIFFile.read(cif_src_path)
    #                 src_atom_array = pdbx.get_structure(src_file, model=1)
    #                 if sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'protein':
    #                     seq_1_to_3 = [_AA1_TO_AA3.get(rt, 'UNK') for rt in list(seq)]
    #                 elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'rna':
    #                     seq_1_to_3 = list(seq)
    #                 elif sample_dict['composition'][sample_dict["cond_c_id"]][0] == 'dna':
    #                     seq_1_to_3 = [f"D{rt}" for rt in list(seq)]
    #                 else:
    #                     seq_1_to_3 = seq.split(' ')

    #                 save_sequence_replaced_array = src_atom_array.copy()
    #                 r_starts = struc.get_residue_starts(save_sequence_replaced_array, add_exclusive_stop=True)
    #                 redesign_id = 0
    #                 for r_s, r_e in zip(r_starts[:-1], r_starts[1:]):
    #                     if save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and self.config.inversefold.data_name != "ligand":
    #                         save_sequence_replaced_array.res_name[r_s:r_e] = seq_1_to_3[redesign_id]
    #                         redesign_id += 1
    #                     elif save_sequence_replaced_array.chain_id[r_s] == sample_dict["cond_c_id"] and self.config.inversefold.data_name == "ligand":
    #                         save_sequence_replaced_array.element[r_s: r_e] = seq_1_to_3
    #                         save_sequence_replaced_array.atom_name[r_s: r_e] = np.char.add(seq_1_to_3, np.array(range(r_e - r_s), dtype=np.str_))
    #                     else: continue

    #                 io.save_structure(out_path, save_sequence_replaced_array)

    #         sample_dict["composition"][sample_dict["cond_c_id"]] = (sample_dict["composition"][sample_dict["cond_c_id"]][0], pred_seqs)
        
        # if self.config.refold.name == "alphafold3":
        #     af3_json = []
        #     for sample_dict in samples_all:
        #         cond_type, pred_seqs = sample_dict["composition"][sample_dict["cond_c_id"]]
        #         for i, p_seq in enumerate(pred_seqs):
        #             name = sample_dict['smp']['title'] + f"_{i+1}"
        #             sequence = []
        #             for c_id, (res_type, seq) in sample_dict['composition'].items():
        #                 if c_id == sample_dict["cond_c_id"]:
        #                     sequence.append({
        #                         res_type: {
        #                             "id": [c_id],
        #                             "sequence": p_seq,
        #                         }
        #                     })
        #                 else:
        #                     sequence.append({
        #                         res_type: {
        #                             "id": [c_id],
        #                             "sequence": seq,
        #                         }
        #                     })
        #             if not self.config.refold.run_data_pipeline:
        #                 for i, s in enumerate(sequence):
        #                     for k, v in s.items():
        #                         if k == 'protein':
        #                             sequence[i][k]["unpairedMsa"] = ""
        #                             sequence[i][k]["pairedMsa"] = ""
        #                             sequence[i][k]["templates"] = []
        #                         elif k == 'rna':
        #                             sequence[i][k]["unpairedMsa"] = ""
        #                         else:
        #                             continue
        #             af3_json.append({
        #                 "name": name,
        #                 "sequences": sequence,
        #                 "modelSeeds": [1],
        #                 "dialect": "alphafold3",
        #                 "version": 1
        #             })
        #     with open(os.path.join(refold_root, "af3_input.json"), 'w') as f:
        #         json.dump(af3_json, f, indent=4)
