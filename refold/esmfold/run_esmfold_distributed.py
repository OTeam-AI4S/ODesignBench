#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import json
import argparse
from typing import List, Dict, Any
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.utils import import_utils as hf_import_utils
import transformers.modeling_utils as hf_modeling_utils
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

# Add parent directory to Python path to allow imports from metrics
# This is now handled by the parent script setting PYTHONPATH, but keep as fallback
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

hf_logging.set_verbosity_error()


def _allow_bin_checkpoint_on_legacy_torch() -> None:
    """
    Allow loading ESMFold `.bin` checkpoints on torch<2.6.
    transformers>=5.3 raises for torch.load(weights_only=True) with `.bin` weights.
    """
    hf_import_utils.check_torch_load_is_safe = lambda: None
    hf_modeling_utils.check_torch_load_is_safe = lambda: None


def setup_distributed(master_port: str = "29500"):
    """Initialize distributed training environment"""
    print("Setting up distributed training...")
    
    # Check if we're in a distributed environment (launched with torchrun)
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        
        print(f"Detected distributed environment: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT={os.environ.get('MASTER_PORT', 'not set')}")
        
        if not dist.is_available():
            raise RuntimeError("Distributed package not available")
        
        try:
            # Initialize distributed environment with timeout
            from datetime import timedelta
            print(f"Rank {rank}: Initializing process group...")
            dist.init_process_group(
                backend="nccl", 
                timeout=timedelta(minutes=30)
            )
            print(f"Rank {rank}: Process group initialized successfully")
            
            # Set device for current process
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                print(f"Rank {rank}: Set CUDA device to {local_rank}")
            
            return local_rank, dist.get_rank(), dist.get_world_size()
        
        except Exception as e:
            print(f"Rank {rank}: Failed to initialize distributed training: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    else:
        # Single GPU mode - no distributed training
        print("No distributed environment detected, running in single GPU mode")
        local_rank = 0
        rank = 0
        world_size = 1
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Using GPU 0 for single GPU processing")
        else:
            print("No CUDA available, using CPU")
        
        return local_rank, rank, world_size


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during distributed cleanup: {e}")
    
    # Clean up environment variables
    for env_var in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        if env_var in os.environ:
            del os.environ[env_var]


def convert_outputs_to_pdb(outputs) -> List[str]:
    """Takes ESMFold outputs and converts them to a list of PDBs (as strings)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def run_esmfold_distributed(
    seq_data: List[Dict[str, str]],
    path_to_esmfold_out: str,
    local_rank: int,
    rank: int,
    world_size: int,
    verbose_gpu: bool = False,
    esmfold_model_dir: str = None,
) -> List[Dict[str, str]]:
    """
    Runs ESMFold in distributed mode and stores results as PDB files.

    Args:
        seq_data: List of dicts with 'name' and 'sequence' keys
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs
        local_rank: Local rank of the current process
        rank: Global rank of the current process
        world_size: Total number of processes
        verbose_gpu: Whether to print detailed GPU memory information

    Returns:
        List of dicts with 'name' and 'pdb_path' keys processed by this rank
    """
    device = f"cuda:{local_rank}"
    
    if verbose_gpu and rank == 0:
        print(f"Starting distributed ESMFold with {len(seq_data)} sequences across {world_size} GPUs")
    
    # Clear GPU memory before loading ESMFold
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose_gpu:
            print(f"Rank {rank}: Cleared GPU cache before ESMFold")
            memory_reserved = torch.cuda.memory_reserved(local_rank) // 1024 // 1024
            memory_allocated = torch.cuda.memory_allocated(local_rank) // 1024 // 1024
            print(f"Rank {rank}: GPU {local_rank} memory before ESMFold: Reserved {memory_reserved}MB, Allocated {memory_allocated}MB")
    
    # Distribute sequences across ranks
    sequences_per_rank = len(seq_data) // world_size
    start_idx = rank * sequences_per_rank
    if rank == world_size - 1:  # Last rank gets remaining sequences
        end_idx = len(seq_data)
    else:
        end_idx = start_idx + sequences_per_rank
    
    rank_seq_data = seq_data[start_idx:end_idx]
    rank_seqs = [item["sequence"] for item in rank_seq_data]
    
    if verbose_gpu:
        print(f"Rank {rank}: Processing {len(rank_seqs)} sequences (indices {start_idx}:{end_idx})")
    
    if len(rank_seqs) == 0:
        return []
    
    # Load ESMFold model
    if esmfold_model_dir is None:
        esmfold_path = os.path.join(os.path.dirname(__file__), "weights")
    else:
        esmfold_path = esmfold_model_dir
    if verbose_gpu and rank == 0:
        print(f"Loading ESMFold model from {esmfold_path}")
    
    _allow_bin_checkpoint_on_legacy_torch()
    tokenizer = AutoTokenizer.from_pretrained(esmfold_path)
    # transformers>=5.3 enforces torch>=2.6 when `weights_only=True` for `.bin` checkpoints.
    # Use `weights_only=False` so ESMFold `.bin` weights work on torch<2.6.
    esm_model = EsmForProteinFolding.from_pretrained(esmfold_path, weights_only=False)
    
    # Move model to GPU
    esm_model = esm_model.to(device)
    
    # Wrap model with DDP only in distributed mode
    if world_size > 1:
        esm_model = DDP(esm_model, device_ids=[local_rank])
        print(f"Rank {rank}: Model wrapped with DDP")
    
    # Determine batch size based on sequence length
    max_nres = max([len(x) for x in rank_seqs])
    if max_nres > 700:
        batch_size = 1
    elif max_nres > 500:
        batch_size = 2
    else:
        batch_size = 4
    
    # Process sequences in batches
    list_of_strings_pdb = []
    num_batches = (len(rank_seqs) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(rank_seqs))
        batch_seqs = rank_seqs[batch_start:batch_end]
        
        if verbose_gpu:
            print(f"Rank {rank}: Processing batch {i+1}/{num_batches} with {len(batch_seqs)} sequences")
        
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        
        # Move inputs to device
        inputs = {k: inputs[k].to(device) for k in inputs}
        
        with torch.no_grad():
            _outputs = esm_model(**inputs)

        _list_of_strings_pdb = convert_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)
    
    # Create output directory if not exists
    if rank == 0:
        os.makedirs(path_to_esmfold_out, exist_ok=True)
    
    # Wait for directory creation only in distributed mode
    if world_size > 1:
        dist.barrier()
    
    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        # Get the name from the sequence data
        item_data = rank_seq_data[i]
        name = item_data["name"]
        fname = f"{name}.pdb"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
        out_esm_paths.append({"name": name, "pdb_path": fdir})
    
    # Clear GPU memory after ESMFold processing
    if torch.cuda.is_available():
        del esm_model
        del tokenizer
        torch.cuda.empty_cache()
        if verbose_gpu:
            print(f"Rank {rank}: Cleared GPU memory after ESMFold processing")
            memory_reserved = torch.cuda.memory_reserved(local_rank) // 1024 // 1024
            memory_allocated = torch.cuda.memory_allocated(local_rank) // 1024 // 1024
            print(f"Rank {rank}: GPU {local_rank} memory after cleanup: Reserved {memory_reserved}MB, Allocated {memory_allocated}MB")
    
    if verbose_gpu:
        print(f"Rank {rank}: ESMFold completed, generated {len(out_esm_paths)} structures")
    
    return out_esm_paths


def main():
    parser = argparse.ArgumentParser(description="Distributed ESMFold processing")
    parser.add_argument("--sequences", type=str, required=True, help="JSON file containing sequences with 'name' and 'sequence' keys")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for PDB files")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port for distributed training")
    parser.add_argument("--esmfold_model_dir", type=str, default=None, help="ESMFold model directory")
    parser.add_argument("--verbose_gpu", action="store_true", help="Enable verbose GPU memory reporting")
    
    args = parser.parse_args()

    
    # Setup distributed training with custom port
    local_rank, rank, world_size = setup_distributed(args.master_port)

    # 将参数写入输出目录中的文件
    if rank == 0:  # 只让rank 0写入文件以避免冲突
        os.makedirs(args.output_dir, exist_ok=True)
        args_file = os.path.join(args.output_dir, "run_args.json")
        args_dict = vars(args)
        with open(args_file, 'w') as f:
            json.dump(args_dict, f, indent=2)
        print(f"Arguments saved to {args_file}")
    # print(args)
    
    try:
        # Load sequences from JSON file
        with open(args.sequences, 'r') as f:
            seq_data = json.load(f)
        
        if rank == 0:
            print(f"Loaded {len(seq_data)} sequences from {args.sequences}")
            print(f"Using {world_size} GPUs for distributed processing")
            print(f"Master port: {args.master_port}")
        
        # Run distributed ESMFold
        out_paths = run_esmfold_distributed(
            seq_data=seq_data,
            path_to_esmfold_out=args.output_dir,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            verbose_gpu=args.verbose_gpu,
            esmfold_model_dir=args.esmfold_model_dir,
        )
        
        # Gather results from all ranks
        if world_size > 1:
            all_out_paths = [None] * world_size
            dist.all_gather_object(all_out_paths, out_paths)
            
            # Save results to file (only rank 0)
            if rank == 0:
                # Flatten the list of lists
                all_paths = []
                for paths in all_out_paths:
                    all_paths.extend(paths)
                
                # Save to JSON file
                results_file = os.path.join(args.output_dir, "esmfold_results.json")
                with open(results_file, 'w') as f:
                    json.dump(all_paths, f, indent=2)
                
                print(f"Generated {len(all_paths)} PDB files")
                print(f"Results saved to {results_file}")
        else:
            # Single GPU mode - save results directly
            results_file = os.path.join(args.output_dir, "esmfold_results.json")
            with open(results_file, 'w') as f:
                json.dump(out_paths, f, indent=2)
            
            print(f"Generated {len(out_paths)} PDB files")
            print(f"Results saved to {results_file}")
    
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        sys.exit(1)
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main() 