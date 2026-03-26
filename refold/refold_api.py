import os
import json
import pickle
import subprocess
import sys
import numpy as np
from pathlib import Path
from time import perf_counter
import biotite.structure as struc
from biotite.structure import BadStructureError
from biotite.structure import info as struc_info
import concurrent.futures
from biotite.structure.io import pdb, pdbx  
from preprocess.ccd_parser import LocalCcdParser
import torch
import copy
import torch.multiprocessing as mp
from typing import Any
from refold.chai1.chai1_distributed_inference import run_folding_on_context
from rdkit.Chem import AllChem as Chem
try:
    from evaluation.metrics.ligand.mol_rec import *
except ImportError:
    print("Warning: Failed to import mol_rec (openbabel dependency). Some features may be unavailable.")
def _chai1_mp_spawn_worker(
    local_rank: int,         # mp.spawn 会自动传入 (rank 0 到 nprocs-1)
    world_size: int,         # 从 args 中传递
    fasta_list: list,        # 从 args 中传递
    output_dir: str,         # 从 args 中传递
    config                   # 从 args 中传递
):
    """
    这是每个被 mp.spawn 启动的独立进程实际执行的函数。
    """
    try:
        print(f"Rank {local_rank}/{world_size}: Starting chai1 inference with {len(fasta_list)} FASTA files")
        
        # 1. 调用你修改后的 chai1 核心函数
        #    它会返回结果 (仅在 rank 0 上) 或 None (在其他 rank 上)
        results = run_folding_on_context(
            local_rank,
            world_size,
            fasta_file_list=fasta_list,
            output_dir=Path(output_dir), # 确保类型正确
            num_diffn_samples=config.refold.num_diffn_samples
        )
        
        # 2. 只有 Rank 0 进程会收到结果，并负责保存
        if local_rank == 0:
            # Ensure output directory exists before saving
            os.makedirs(output_dir, exist_ok=True)
            
            # Always save results, even if empty, to help with debugging
            output_file = os.path.join(output_dir, 'chai1_cands.pkl')
            if results:
                print(f"Rank 0: Received {len(results)} results. Saving...")
                with open(output_file, 'wb') as f:
                    pickle.dump(results, f)
                print(f"Rank 0: Results saved to {output_file}")
            else:
                print("Rank 0: Warning - No results were returned. Saving empty list for debugging...")
                # Save empty list to help debug why no results were generated
                with open(output_file, 'wb') as f:
                    pickle.dump([], f)
                print(f"Rank 0: Empty results saved to {output_file}")
        else:
            print(f"Rank {local_rank}: Completed (results only saved on rank 0)")
    except Exception as e:
        print(f"Rank {local_rank}: Error during chai1 inference: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to ensure the error is visible
        raise

from biotite.structure import get_residue_starts
from biotite.structure import BondType
from biotite.interface.rdkit import to_mol
from biotite.structure import BondList
RDKIT_TO_BIOTITE_BOND_TYPE = {
    Chem.BondType.UNSPECIFIED: BondType.ANY,
    Chem.BondType.SINGLE: BondType.SINGLE,
    Chem.BondType.DOUBLE: BondType.DOUBLE,
    Chem.BondType.TRIPLE: BondType.TRIPLE,
    Chem.BondType.QUADRUPLE: BondType.QUADRUPLE,
    Chem.BondType.DATIVE: BondType.COORDINATION,
    # [Yuanle] 以上为biotite定义的映射，额外添加AROMATIC的映射
    Chem.BondType.AROMATIC: BondType.AROMATIC,
}

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
def get_nucleic_acid_sequence(chain_struct):
        """get seq from structure"""
        res_starts = get_residue_starts(chain_struct, add_exclusive_stop=True)
        sequence = ""

        for res_start, res_end in zip(res_starts[:-1], res_starts[1:]):
            res = chain_struct[res_start:res_end]
            try:
                sequence += NA_STD_RESIDUES_RES_NAME_TO_ONE[res[0].res_name]
            except:
                continue
        
        return sequence if sequence else None

def _gpus_to_str(gpus) -> str:
    """Normalize gpus to comma-separated string (e.g. '5,6,7'). Handles list, string '5,6,7', or \"['5','6','7']\"."""
    def _clean(x):
        return str(x).strip().strip("[]'\"")
    if isinstance(gpus, (list, tuple)):
        return ",".join(_clean(x) for x in gpus)
    s = str(gpus).strip()
    if s.startswith("["):
        import re
        s = re.sub(r"[\s\[\]'\"]", "", s)
    parts = [x.strip().strip("[]'\"") for x in s.split(",") if x.strip()]
    return ",".join(parts) if parts else _clean(s)


class ReFold:

    def __init__(self, config):

        self.config = config

    def _make_result(
        self,
        stage: str,
        success: bool = True,
        outputs: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "success": bool(success),
            "stage": stage,
            "outputs": outputs or {},
            "details": details or {},
        }

    @staticmethod
    def _count_files(root_dir: str, patterns: list[str]) -> int:
        root = Path(root_dir)
        if not root.exists():
            return 0
        count = 0
        for pattern in patterns:
            count += len(list(root.rglob(pattern)))
        return count

    def run(self, action: str, **kwargs) -> dict[str, Any]:
        dispatch = {
            "run_alphafold3": self.run_alphafold3,
            "run_chai1": self.run_chai1,
            "chai1": self.run_chai1,
            "run_esmfold": self.run_esmfold,
            "esmfold": self.run_esmfold,
            "make_esmfold_json_multi_process": self.make_esmfold_json_multi_process,
            "prepare_esmfold_json": self.make_esmfold_json_multi_process,
            "make_af3_json_multi_process": self.make_af3_json_multi_process,
            "prepare_af3_json": self.make_af3_json_multi_process,
            "make_af3_json_pbp_target_msa": self.make_af3_json_pbp_target_msa,
            "make_chai1_fasta_multi_process": self.make_chai1_fasta_multi_process,
            "prepare_chai1_fasta": self.make_chai1_fasta_multi_process,
        }
        if action not in dispatch:
            known = ", ".join(sorted(dispatch.keys()))
            raise ValueError(f"Unknown refold action '{action}'. Supported: {known}")
        t0 = perf_counter()
        result = dispatch[action](**kwargs)
        elapsed = perf_counter() - t0
        if isinstance(result, dict):
            details = result.setdefault("details", {})
            details["elapsed_seconds"] = round(elapsed, 3)
        print(f"[timing] refold.{action}: {elapsed:.2f}s")
        return result

    def run_alphafold3(self, input_json: str, output_dir: str) -> dict[str, Any]:
        output_dir = str(Path(output_dir).resolve())
        input_json = str(Path(input_json).resolve())
        dialect_fix = self._normalize_af3_input_dialect(input_json)
        gpus_str = _gpus_to_str(self.config.gpus)
        cmd = ["bash", f"{self.config.refold.af3_exec}", f"{self.config.refold.exp_name}", f"{input_json}", f"{output_dir}", gpus_str, f"{self.config.refold.run_data_pipeline}", f"{self.config.refold.cache_dir}"]
        subprocess.run(cmd, check=True)
        # Verify output was produced
        cif_files = list(Path(output_dir).rglob("*.cif"))
        if not cif_files:
            log_dir = Path(output_dir).parent / "af3_log"
            log_hint = f" Check {log_dir}/ for AF3 GPU logs." if log_dir.exists() else ""
            raise FileNotFoundError(
                f"AF3 completed but no CIF files in {output_dir}.{log_hint} "
                "Possible causes: (1) Container writes to a different path; "
                "(2) AF3 failed - check af3_log/*.log; "
                "(3) templatesPath in af3_input.json - ensure backbone PDB paths are accessible inside container."
            )
        return self._make_result(
            stage="refold.run_alphafold3",
            outputs={"input_json": str(input_json), "output_dir": str(output_dir)},
            details={"num_cif_files": len(cif_files), **dialect_fix},
        )

    def _normalize_af3_input_dialect(self, input_json: str) -> dict[str, Any]:
        """
        Ensure AF3 input JSON dialect matches the expected format.
        Normalizes stale input files in-place to avoid runtime failure.
        Default dialect is 'alphafoldserver' (AlphaFold Server API format).
        """
        expected = str(getattr(self.config.refold, "af3_input_dialect", "alphafoldserver"))
        try:
            with open(input_json, "r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: Failed to inspect AF3 input dialect for {input_json}: {exc}")
            return {
                "af3_expected_dialect": expected,
                "af3_dialect_normalized": False,
                "af3_dialect_updates": 0,
            }

        jobs = payload if isinstance(payload, list) else [payload]
        updates = 0
        for job in jobs:
            if not isinstance(job, dict):
                continue
            current = str(job.get("dialect", "")).strip()
            if current != expected:
                job["dialect"] = expected
                updates += 1

        if updates > 0:
            with open(input_json, "w") as f:
                json.dump(payload, f, indent=4)
            print(
                f"[refold] Normalized AF3 input dialect to '{expected}' for "
                f"{updates} job(s): {input_json}"
            )

        return {
            "af3_expected_dialect": expected,
            "af3_dialect_normalized": updates > 0,
            "af3_dialect_updates": updates,
        }
    
    def run_chai1(self, fasta_list: list, output_dir: str) -> dict[str, Any]:
        
        # 1. 检查 GPU 可用性（必须在解析 CUDA_VISIBLE_DEVICES 之前）
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. Chai1 requires GPUs to run.")
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
            raise RuntimeError("CUDA is not available. Chai1 requires GPUs to run.")
        
        # 2. 确定要使用的 GPU 数量
        # 首先检查 CUDA_VISIBLE_DEVICES 环境变量（如果设置了）
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        actual_gpu_count = torch.cuda.device_count()
        
        if cuda_visible:
            # 解析环境变量中的 GPU 列表
            gpu_list = [gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()]
            world_size = len(gpu_list)
            print(f"CUDA_VISIBLE_DEVICES set to: {cuda_visible}")
            print(f"  Requested GPUs: {world_size}")
            print(f"  Actually available GPUs: {actual_gpu_count}")
            
            # 验证实际可用的 GPU 数量
            if actual_gpu_count == 0:
                print("Error: CUDA_VISIBLE_DEVICES is set but no GPUs are actually available.")
                print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
                print(f"  torch.cuda.device_count(): {actual_gpu_count}")
                raise RuntimeError("CUDA_VISIBLE_DEVICES is set but no GPUs are actually available.")
            
            # 确保请求的 GPU 数量不超过实际可用数量
            if world_size > actual_gpu_count:
                print(f"Warning: Requested {world_size} GPUs but only {actual_gpu_count} are available.")
                print(f"  Using {actual_gpu_count} GPUs instead.")
                world_size = actual_gpu_count
        else:
            # 如果没有设置环境变量，使用 torch 检测
            world_size = actual_gpu_count
            print(f"CUDA_VISIBLE_DEVICES not set. Using all available GPUs: {world_size} GPUs")
        
        if world_size == 0:
            print("Error: No GPUs found for distributed folding.")
            print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible if cuda_visible else 'not set'}")
            print(f"  torch.cuda.device_count(): {actual_gpu_count}")
            raise RuntimeError("No GPUs found for distributed folding.")
            
        print(f"Found {world_size} GPUs. Spawning processes...")

        # 2. (关键) 设置主进程的环境变量
        #    这必须在 mp.spawn 之前完成，以便所有子进程都能继承它们
        #    如果环境变量已经设置（例如由run_chai1_gpu.py设置），则使用已有的值
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355" # 默认端口，如果未设置则使用此端口
        
        # 3. 准备要传递给 worker 函数的参数
        #    注意: torch.multiprocessing.spawn 只自动传递 rank (local_rank)
        #    world_size 需要手动传递
        args_to_pass = (
            world_size,      # 需要手动传递，因为 spawn 不会自动传递
            fasta_list, 
            output_dir, 
            self.config
        )
        
        # 4. 启动！
        mp.spawn(
            _chai1_mp_spawn_worker, # 你在上面定义的 worker 函数
            args=args_to_pass,        # 要传递的参数
            nprocs=world_size,        # 启动的进程数
            join=True                 # 阻塞主进程，直到所有子进程完成
        )
        
        print(f"Chai1 distributed folding complete. Results saved in {output_dir}")
        candidate_pickle = Path(output_dir) / "chai1_cands.pkl"
        num_candidates = 0
        if candidate_pickle.exists():
            try:
                with open(candidate_pickle, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, list):
                    num_candidates = len(payload)
            except Exception as e:
                print(f"Warning: Failed to parse chai1 results from {candidate_pickle}: {e}")
        # 注意：原始的 pickle.dump(...) 已被移入 worker 函数中
        return self._make_result(
            stage="refold.run_chai1",
            outputs={"output_dir": str(output_dir)},
            details={
                "num_fasta": len(fasta_list),
                "world_size": world_size,
                "num_candidates": num_candidates,
            },
        )

    def run_esmfold(self, sequences_file_json: str, output_dir: str) -> dict[str, Any]:
        sequences_file_json = str(Path(sequences_file_json).resolve())
        output_dir = str(Path(output_dir).resolve())
        esmfold_model_dir = str(Path(self.config.refold.esmfold_model_dir).resolve())
        esmfold_script = os.path.join(os.path.dirname(__file__), "esmfold", "run_esmfold.sh")
        # breakpoint()
        cmd = [
            esmfold_script,
            # "-n", "batch_esmfold",
            "-s", sequences_file_json,
            "-o", output_dir,
            "-g", str(len(_gpus_to_str(self.config.gpus).split(","))),
            "-p", str(self.config.refold.master_port),
            "-m", esmfold_model_dir
        ]

        # breakpoint()
        
        print(f"Running ESMFold command: {' '.join(cmd)}")
        
        try:
            env = os.environ.copy()
            # Keep ESMFold subprocess on the same interpreter as current process.
            env["PYTHON"] = sys.executable
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            print(f"ESMFold execution completed successfully")
            if result.stdout:
                print("STDOUT:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"ESMFold execution failed: {e}")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            raise e
        output_count = self._count_files(output_dir, ["*.pdb", "*.cif"])
        return self._make_result(
            stage="refold.run_esmfold",
            outputs={"input_json": str(sequences_file_json), "output_dir": str(output_dir)},
            details={"num_outputs": output_count},
        )

    # @staticmethod
    # def get_smiles_from_ligand(chain_atom_array):
    #     """
    #     Extract SMILES from ligand structure using RDKit.
    #     Attempts to generate SMILES from 3D coordinates.
        
    #     Args:
    #         chain_atom_array: Biotite AtomArray containing ligand atoms
            
    #     Returns:
    #         str: SMILES string representation of the ligand
    #     """
    #     try:
    #         from rdkit import Chem
    #         from rdkit.Chem import AllChem
            
    #         # Create RDKit molecule from atom array
    #         mol = Chem.RWMol()
    #         atom_indices = {}
            
    #         # Add atoms
    #         for i, atom in enumerate(chain_atom_array):
    #             element = atom.element
    #             rdkit_atom = Chem.Atom(element)
    #             idx = mol.AddAtom(rdkit_atom)
    #             atom_indices[i] = idx
            
    #         # Try to infer bonds from distances
    #         coords = chain_atom_array.coord
    #         for i in range(len(chain_atom_array)):
    #             for j in range(i + 1, len(chain_atom_array)):
    #                 dist = np.linalg.norm(coords[i] - coords[j])
    #                 # Simple distance-based bonding (can be improved)
    #                 if dist < 1.8:  # Typical bond length threshold
    #                     mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
            
    #         # Convert to standard molecule
    #         mol = mol.GetMol()
    #         Chem.SanitizeMol(mol)
            
    #         # Generate SMILES
    #         smiles = Chem.MolToSmiles(mol)
    #         return smiles
            
    #     except Exception as e:
    #         print(f"Warning: Failed to generate SMILES from structure: {e}")
    #         # Return a placeholder or raise error
    #         return None
    
    @staticmethod
    def get_smiles_from_ligand(gen_ligand):
        """
        从 3D 坐标重建配体并生成 SMILES。
        关键改进：确保生成的 SMILES 经过 RDKit 验证，避免无效 SMILES 导致 Chai1 崩溃。
        """
        gen_lig_atom_index = list(range(len(gen_ligand)))
        gen_lig_positions = copy.deepcopy(gen_ligand.coord).astype(float)
        gen_lig_atom_types = [map_atom_symbol_to_atomic_number(atom_type) for atom_type in gen_ligand.element]
        
        # 尝试使用 basic_mode=False 以获得更准确的键级（Bond Order）
        # 如果失败，回退到 basic_mode=True
        rd_mol = None
        try:
            rd_mol = reconstruct_mol(
                gen_lig_positions,
                gen_lig_atom_types,
                basic_mode=False,  # 使用更精确的模式
            )
        except:
            # 如果失败，回退到 basic_mode
            try:
                rd_mol = reconstruct_mol(
                    gen_lig_positions,
                    gen_lig_atom_types,
                    basic_mode=True,
                )
            except Exception as e:
                print(f"警告：无法从 3D 坐标重建配体分子: {e}")
                return None
        
        if rd_mol is None:
            return None
        
        rec_lig_bond = np.array([
            (
                gen_lig_atom_index[bond.GetBeginAtomIdx()],
                gen_lig_atom_index[bond.GetEndAtomIdx()],
                RDKIT_TO_BIOTITE_BOND_TYPE[bond.GetBondType()]
            )
            for bond in rd_mol.GetBonds()
        ])

        gen_ligand.bonds = BondList(
            gen_ligand.array_length(),
            rec_lig_bond
        )
        mol = to_mol(gen_ligand)
        smiles = ReFold.get_roundtrip_smiles(mol)
        
        # 最终验证：确保生成的 SMILES 可以被 RDKit 解析
        if smiles is not None:
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is None:
                print(f"警告：生成的 SMILES '{smiles}' 无法被 RDKit 解析，已丢弃")
                return None
        
        return smiles
    
    @staticmethod
    def get_roundtrip_smiles(mol):
        """
        生成 SMILES 并验证其有效性。
        确保生成的 SMILES 可以被 RDKit 正确解析，避免 Chai1 崩溃。
        """
        try:
            smiles = Chem.MolToSmiles(mol, allBondsExplicit=True, canonical=False)
            # 强制验证：确保生成的 SMILES 可以被 RDKit 解析
            test_mol = Chem.MolFromSmiles(smiles)
            if test_mol is not None:
                # 再次生成 canonical SMILES，确保兼容性
                try:
                    canonical_smiles = Chem.MolToSmiles(test_mol, isomericSmiles=True)
                    return canonical_smiles
                except:
                    return smiles  # 如果生成 canonical 失败，返回原始 SMILES
            else:
                # 如果无法解析，尝试修复
                test_mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if test_mol is not None:
                    try:
                        Chem.SanitizeMol(test_mol)
                        return Chem.MolToSmiles(test_mol, isomericSmiles=True)
                    except:
                        return None
                return None
        except Exception as e:
            print(f"警告：生成 SMILES 时出错: {e}")
            return None
    
    @staticmethod
    def make_af3_json_from_backbone(
        backbone_path: Path,
        run_data_pipeline: bool,
        unpaired_msa_cache: dict | None = None,
        paired_msa_cache: dict | None = None,
        template_cache: dict | None = None,
        use_backbone_as_template: bool = False,
        dialect: str = "alphafoldserver",
        *,
        # PBP-specific: inject only the target chain's MSA into the AF3 input.
        # Binder's MSA is intentionally excluded (set to empty strings).
        target_chain_id: str | None = None,
        target_unpaired_msa_path: str | None = None,
    ):
        """
        Build AF3 input JSON from backbone PDB/CIF (sequences only).
        use_backbone_as_template is ignored: the AF3 container does not support custom templates
        (templatesPath is rejected). Refold is sequence-based only (no CDR-only or scaffold fixing).

        Server API dialect (alphafoldserver) key format:
          proteinChain, rnaSequence, dnaSequence, ligand, ion
        AF3 native dialect (alphafold3) key format:
          protein, rna, dna, ligand (with full nested dicts)
        """
        is_server = dialect == "alphafoldserver"
        single_input = {
            "name": backbone_path.stem,
            "sequences": [],
            "modelSeeds": [1],
            "dialect": dialect,
            "version": 1
        }
        if backbone_path.suffix == '.cif':
            cif_file = pdbx.CIFFile.read(backbone_path)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:
            pdb_file = pdb.PDBFile.read(backbone_path)
            atom_array = pdb.get_structure(pdb_file, model=1)
        chain_ids = np.unique(atom_array.chain_id)
        for chain_id in chain_ids:
            chain_atom_array = atom_array[atom_array.chain_id == chain_id]
            if chain_atom_array.hetero.all():
                ccdCode = chain_atom_array.res_name[0].upper()

                # designed ligand (-L): use SMILES
                if ccdCode == '-L':
                    smiles = ReFold.get_smiles_from_ligand(chain_atom_array)
                    if smiles:
                        if is_server:
                            # Server API does not support SMILES-based ligands.
                            # Emit as AF3 native format with ccdCodes (won't parse
                            # but preserves SMILES in output for debugging).
                            print(
                                f"Warning: SMILES ligand in chain {chain_id} is not "
                                f"supported in alphafoldserver dialect. "
                                f"SMILES='{smiles}'. Skipping this entity."
                            )
                        else:
                            # AF3 native: ligand dict with smiles + id
                            single_input["sequences"].append({
                                "ligand": {
                                    "smiles": smiles,
                                    "id": [str(chain_id)]
                                }
                            })
                    else:
                        print(f"Warning: Failed to generate SMILES for designed ligand in chain {chain_id}")
                    continue

                # standard ligand / ion
                if is_server:
                    # Server API: ligand value is a plain string ("CCD_Xxx" or ion name)
                    if ccdCode in ('NA', 'K', 'MG', 'CA', 'ZN', 'MN', 'FE', 'FE2', 'CO', 'CU', 'CU1'):
                        ligand_seq = ccdCode  # ion
                    else:
                        ligand_seq = f"CCD_{ccdCode}"
                    single_input["sequences"].append({"ligand": ligand_seq})
                else:
                    # AF3 native: ligand dict with ccdCodes (list) + id
                    single_input["sequences"].append({
                        "ligand": {
                            "ccdCodes": [ccdCode],
                            "id": [str(chain_id)]
                        }
                    })

            elif np.isin(chain_atom_array.res_name, ['DA', 'DC', 'DG', 'DT', 'DN']).all():
                sequence = get_nucleic_acid_sequence(chain_atom_array)
                if is_server:
                    # Server API dialect: dnaSequence only accepts 'sequence'
                    single_input["sequences"].append({"dnaSequence": {"sequence": sequence}})
                else:
                    single_input["sequences"].append({
                        "dna": {"sequence": sequence, "id": [str(chain_id)]}
                    })

            elif np.isin(chain_atom_array.res_name, ['A', 'C', 'G', 'U', 'N']).all():
                sequence = get_nucleic_acid_sequence(chain_atom_array)
                if is_server:
                    # Server API dialect: rnaSequence only accepts 'sequence'
                    single_input["sequences"].append({"rnaSequence": {"sequence": sequence}})
                else:
                    s = {"rna": {"sequence": sequence, "id": [str(chain_id)]}}
                    if not run_data_pipeline:
                        s["rna"]["unpairedMsa"] = ""
                    single_input["sequences"].append(s)

            else:
                try:
                    sequence = str(struc.to_sequence(chain_atom_array, allow_hetero=True)[0][0])
                except BadStructureError:
                    standard_aa = set(struc_info.amino_acid_names())
                    res_names = chain_atom_array.res_name.copy()
                    non_std = ~np.isin(res_names, list(standard_aa))
                    if non_std.any():
                        res_names[non_std] = "GLY"
                        chain_copy = chain_atom_array.copy()
                        chain_copy.res_name = res_names
                        sequence = str(struc.to_sequence(chain_copy, allow_hetero=True)[0][0])
                    else:
                        raise

                if is_server:
                    # Server API dialect: proteinChain accepts 'sequence' + MSA fields (with patch).
                    # The AF3_DIALECT_PATCH in run_af3.sh extends from_alphafoldserver_dict
                    # to also read unpairedMsaPath/pairedMsaPath/templates fields.
                    pc = {"sequence": sequence}
                    if not run_data_pipeline:
                        if target_chain_id is not None:
                            if str(chain_id) == str(target_chain_id) and target_unpaired_msa_path:
                                pc["unpairedMsaPath"] = str(target_unpaired_msa_path)
                            else:
                                pc["unpairedMsa"] = ""
                            pc["pairedMsa"] = ""
                            pc["templates"] = []
                        else:
                            if unpaired_msa_cache and sequence in unpaired_msa_cache:
                                pc["unpairedMsaPath"] = unpaired_msa_cache[sequence]
                            else:
                                pc["unpairedMsa"] = ""
                            if paired_msa_cache and sequence in paired_msa_cache:
                                pc["pairedMsaPath"] = paired_msa_cache[sequence]
                            else:
                                pc["pairedMsa"] = ""
                            pc["templates"] = []
                    s = {"proteinChain": pc}
                else:
                    s = {"protein": {"sequence": sequence, "id": [str(chain_id)]}}
                    if not run_data_pipeline:
                        if target_chain_id is not None:
                            if str(chain_id) == str(target_chain_id) and target_unpaired_msa_path:
                                s["protein"]["unpairedMsaPath"] = str(target_unpaired_msa_path)
                            else:
                                s["protein"]["unpairedMsa"] = ""
                            s["protein"]["pairedMsa"] = ""
                            s["protein"]["templates"] = []
                        else:
                            if unpaired_msa_cache and sequence in unpaired_msa_cache:
                                s["protein"]["unpairedMsaPath"] = unpaired_msa_cache[sequence]
                            else:
                                s["protein"]["unpairedMsa"] = ""
                            if paired_msa_cache and sequence in paired_msa_cache:
                                s["protein"]["pairedMsaPath"] = paired_msa_cache[sequence]
                            else:
                                s["protein"]["pairedMsa"] = ""
                            s["protein"]["templates"] = []
                single_input["sequences"].append(s)
        return single_input
    
    @staticmethod
    def make_esmfold_json_from_backbone(backbone_path: Path,):
        single_input = {
            "name": backbone_path.stem,
            "sequence": "",
        }
        if backbone_path.suffix == '.cif':
            cif_file = pdbx.CIFFile.read(backbone_path)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:
            pdb_file = pdb.PDBFile.read(backbone_path)
            atom_array = pdb.get_structure(pdb_file, model=1)
        chain_ids = np.unique(atom_array.chain_id)
        # select the first chain only
        chain_id = chain_ids[0]
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]  
        sequence = str(struc.to_sequence(chain_atom_array)[0][0])
        single_input["sequence"] = sequence
        return single_input

    def make_esmfold_json_multi_process(self, backbone_dir: str, output_dir: str):
        
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        if len(backbone_path_list) == 0:
            print(f"Warning: No backbone PDB files found in {backbone_dir}, try cif format.")
            backbone_path_list = list(Path(backbone_dir).glob("*.cif"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = []
            for backbone_path in backbone_path_list:
                future = executor.submit(self.make_esmfold_json_from_backbone, backbone_path)
                futures.append(future)
            af3_input_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        with open(output_dir, 'w') as f:
            json.dump(af3_input_list, f, indent=4)
        return self._make_result(
            stage="refold.make_esmfold_json_multi_process",
            outputs={"output_json": str(output_dir)},
            details={"num_backbones": len(backbone_path_list)},
        )
    
    def make_af3_json_multi_process(self, backbone_dir: str, output_path: str, use_backbone_as_template: bool = False):
        """
        Build AF3 input JSON from all backbones in backbone_dir (sequence-only; no template).
        use_backbone_as_template is ignored (AF3 container does not support custom templates).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        if len(backbone_path_list) == 0:
            print(f"Warning: No backbone PDB files found in {backbone_dir}, try cif format.")
            backbone_path_list = list(Path(backbone_dir).glob("*.cif"))
        def _load_json_opt(attr_name):
            try:
                p = getattr(self.config.refold, attr_name, None)
                if p and os.path.isfile(p):
                    with open(p) as f:
                        return json.load(f)
            except (AttributeError, OSError, json.JSONDecodeError):
                pass
            return None
        unpaired = _load_json_opt('unpaired_msa_cache')
        paired = _load_json_opt('paired_msa_cache')
        tmpl = _load_json_opt('template_cache')
        dialect = str(getattr(self.config.refold, "af3_input_dialect", "alphafoldserver"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = []
            for backbone_path in backbone_path_list:
                future = executor.submit(
                    self.make_af3_json_from_backbone, backbone_path, self.config.refold.run_data_pipeline,
                    unpaired_msa_cache=unpaired, paired_msa_cache=paired, template_cache=tmpl,
                    use_backbone_as_template=use_backbone_as_template, dialect=dialect
                )
                futures.append(future)
            af3_input_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        with open(output_path, 'w') as f:
            json.dump(af3_input_list, f, indent=4)
        return self._make_result(
            stage="refold.make_af3_json_multi_process",
            outputs={"output_json": str(output_path)},
            details={"num_backbones": len(backbone_path_list)},
        )

    def make_af3_json_pbp_target_msa(
        self,
        backbone_dir: str,
        output_path: str,
        pbp_info_df: object,
        use_backbone_as_template: bool = False,
    ) -> dict[str, Any]:
        """
        Build AF3 input JSON for PBP with the following rule:
        - provide only target chain's MSA (unpaired MSA)
        - exclude templates (templates: [])
        - exclude binder's MSA (unpairedMsa/pairedMsa are empty)
        """
        if self.config.refold.run_data_pipeline:
            raise ValueError(
                "PBP target-MSA mode requires refold.run_data_pipeline=false, "
                "otherwise AF3 will run its own MSA/template pipeline and violate the requirement."
            )

        from inversefold.pbp_csv_utils import match_pdb_to_pbp_info

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        if len(backbone_path_list) == 0:
            print(f"Warning: No backbone PDB files found in {backbone_dir}, try cif format.")
            backbone_path_list = list(Path(backbone_dir).glob("*.cif"))
        if len(backbone_path_list) == 0:
            raise FileNotFoundError(f"No backbones found under {backbone_dir}")

        repo_root = Path(__file__).resolve().parents[1]
        pbp_msa_root = repo_root / "assets" / "pbp" / "msa"
        pbp_msa_root = pbp_msa_root.resolve()
        if not pbp_msa_root.exists():
            raise FileNotFoundError(f"PBP MSA root not found: {pbp_msa_root}")

        # Prefer the canonical target id list from target_config.csv when available.
        target_ids: list[str] = []
        target_cfg = repo_root / "assets" / "pbp" / "config" / "target_config.csv"
        if target_cfg.exists():
            import csv

            with target_cfg.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tid = (row.get("target_id") or "").strip()
                    if tid:
                        target_ids.append(tid)
        if not target_ids:
            target_ids = sorted([p.name for p in pbp_msa_root.iterdir() if p.is_dir()])
        target_ids = sorted(set(target_ids))

        def read_a3m_query_sequence(a3m_path: Path) -> str:
            with a3m_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith(">"):
                        query = next(handle, "").strip()
                        if query:
                            return query
                        break
            raise ValueError(f"Failed to read query sequence from A3M: {a3m_path}")

        def infer_target_id(stem: str) -> str:
            # Most backbones contain target_id as a prefix or substring. Use longest-match.
            stem_lower = stem.lower()
            for tid in sorted(target_ids, key=len, reverse=True):
                if tid and (tid.lower() == stem_lower or tid.lower() in stem_lower):
                    return tid
            # Fallback: split by common delimiters.
            for sep in ["-", "_"]:
                if sep in stem:
                    return stem.split(sep, 1)[0]
            return stem

        def validate_target_msa(single_input: dict[str, Any], target_chain_id: str, msa_path: Path, target_id: str) -> None:
            query_sequence = read_a3m_query_sequence(msa_path).strip()
            target_sequence = None
            dialect = single_input.get("dialect", "alphafoldserver")
            is_server = dialect == "alphafoldserver"
            for seq in single_input.get("sequences", []):
                # AF3 native dialect: {"protein": {"sequence": ..., "id": [...]}}
                protein = seq.get("protein")
                if protein:
                    chain_ids = [str(item) for item in protein.get("id", [])]
                    if str(target_chain_id) in chain_ids:
                        target_sequence = str(protein.get("sequence", "")).strip()
                        break
                # AF3 server dialect: {"proteinChain": {"sequence": ...}}
                # proteinChain has no "id" field; chain ordering comes from the input PDB.
                # Validate by checking that SOME proteinChain entry matches the MSA query.
                protein_chain = seq.get("proteinChain")
                if protein_chain and is_server:
                    candidate = str(protein_chain.get("sequence", "")).strip()
                    if candidate == query_sequence:
                        target_sequence = candidate
                        break
            if query_sequence != target_sequence:
                raise ValueError(
                    "PBP target MSA query does not match the selected target chain sequence: "
                    f"backbone='{single_input.get('name', '')}', target_id='{target_id}', target_chain='{target_chain_id}', "
                    f"msa='{msa_path}', query_len={len(query_sequence)}, chain_len={len(target_sequence)}"
                )

        def _build_one(backbone_path: Path) -> dict[str, Any]:
            pbp_info = match_pdb_to_pbp_info(backbone_path, pbp_info_df) if pbp_info_df is not None else None
            if pbp_info_df is not None and pbp_info is None:
                raise ValueError(
                    f"No PBP CSV info found for backbone '{backbone_path.name}'. "
                    "Ensure pbp_info.csv design_name matches formatted designs and backbone-derived names."
                )
            target_chain_id = pbp_info["target_chain"] if pbp_info is not None else "B"
            dialect = str(getattr(self.config.refold, "af3_input_dialect", "alphafoldserver"))

            target_id = ""
            if pbp_info is not None:
                target_id = str(pbp_info.get("target_id", "")).strip()
            if not target_id:
                target_id = infer_target_id(backbone_path.stem)
            msa_path = pbp_msa_root / target_id / "non_pairing.a3m"
            if not msa_path.exists():
                raise FileNotFoundError(
                    f"Missing PBP target MSA for target_id='{target_id}': {msa_path}. "
                    "Please ensure assets/pbp/msa/<target_id>/non_pairing.a3m exists."
                )
            # Inside the AF3 Docker container, /assets is mounted from AF3_ASSETS (the repo's assets/).
            # Keep the host path for reading; pass container path to AF3 JSON.
            host_msa_path = msa_path
            container_msa_path = f"/assets/pbp/msa/{target_id}/non_pairing.a3m"

            single_input = self.make_af3_json_from_backbone(
                backbone_path,
                self.config.refold.run_data_pipeline,
                unpaired_msa_cache=None,
                paired_msa_cache=None,
                template_cache=None,
                use_backbone_as_template=use_backbone_as_template,
                dialect=dialect,
                target_chain_id=target_chain_id,
                target_unpaired_msa_path=container_msa_path,
            )
            validate_target_msa(single_input, target_chain_id, host_msa_path, target_id)
            return single_input

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = [executor.submit(_build_one, bp) for bp in backbone_path_list]
            af3_input_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        with open(output_path, "w") as f:
            json.dump(af3_input_list, f, indent=4)

        return self._make_result(
            stage="refold.make_af3_json_pbp_target_msa",
            outputs={"output_json": str(output_path)},
            details={
                "num_backbones": len(backbone_path_list),
                "pbp_msa_root": str(pbp_msa_root),
            },
        )

    @staticmethod
    def make_chai1_fasta_from_backbone(backbone_path: Path, ccd_parser: LocalCcdParser, output_path: str, reference_pdb_path: Path = None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Implement the logic to create a FASTA file from the backbone structure
        if backbone_path.suffix == '.cif':
            cif_file = pdbx.CIFFile.read(backbone_path)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:
            pdb_file = pdb.PDBFile.read(backbone_path)
            atom_array = pdb.get_structure(pdb_file, model=1)
        chain_ids = np.unique(atom_array.chain_id)
        fasta_strings = []
        
        # 用于跟踪已添加的配体，避免重复
        processed_ligands = set()
        # 用于收集所有蛋白质链的序列
        protein_sequences = []
        
        # 首先处理蛋白质、DNA、RNA序列和配体
        for chain_id in chain_ids:
            chain_atom_array = atom_array[atom_array.chain_id == chain_id]
            if chain_atom_array.hetero.any():
                ccdCode = chain_atom_array.res_name[0].upper()
                
                # 跳过已处理的配体
                if ccdCode in processed_ligands:
                    continue
                
                # 处理设计的配体 (-L)
                if ccdCode == '-L':
                    # 从 3D 结构生成 SMILES
                    smiles = ReFold.get_smiles_from_ligand(chain_atom_array)
                    if smiles is None:
                        print(f"Warning: Failed to generate SMILES from structure for designed ligand (-L)")
                        continue
                    # 最终验证：确保生成的 SMILES 可以被 RDKit 解析
                    test_mol = Chem.MolFromSmiles(smiles)
                    if test_mol is None:
                        print(f"Warning: Generated SMILES '{smiles}' for -L cannot be parsed by RDKit, skipping")
                        continue
                    fasta_strings.append(f">ligand|name=-L\n{smiles}\n")
                    processed_ligands.add(ccdCode)
                else:
                    # 处理已知配体（从 CCD 获取）
                    smiles_result = ccd_parser.get_smiles(ccdCode)
                    if smiles_result is None:
                        print(f"Warning: Failed to get SMILES for {ccdCode}")
                        continue
                    # Handle both list and string return types
                    if isinstance(smiles_result, list):
                        smiles = smiles_result[0]
                    else:
                        smiles = smiles_result
                    # Remove quotes if present (虽然现在应该已经处理过了)
                    smiles = smiles.strip('"\'')
                    # 最终验证：确保从 CCD 获取的 SMILES 可以被 RDKit 解析
                    test_mol = Chem.MolFromSmiles(smiles)
                    if test_mol is None:
                        print(f"Warning: SMILES '{smiles}' for {ccdCode} cannot be parsed by RDKit, skipping")
                        continue
                    fasta_strings.append(f">ligand|name={ccdCode}\n{smiles}\n")
                    processed_ligands.add(ccdCode)
            elif np.isin(chain_atom_array.res_name, ['DA', 'DC', 'DG', 'DT']).all():
                sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                fasta_strings.append(f">dna|name=dna\n{sequence}\n")
            elif np.isin(chain_atom_array.res_name, ['A', 'C', 'G', 'U']).all():
                sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                fasta_strings.append(f">rna|name=rna\n{sequence}\n")
            else:
                # 收集蛋白质序列，稍后合并
                sequence = str(struc.to_sequence(chain_atom_array)[0][0])
                protein_sequences.append(sequence)
        
        # 合并所有蛋白质链的序列，只添加一次
        if protein_sequences:
            combined_protein_sequence = ''.join(protein_sequences)
            fasta_strings.insert(0, f">protein|name=protein\n{combined_protein_sequence}\n")
        
        # 从参考PDB文件中提取离子和配体信息
        if reference_pdb_path is not None and reference_pdb_path.exists():
            try:
                ref_pdb_file = pdb.PDBFile.read(str(reference_pdb_path))
                ref_atom_array = pdb.get_structure(ref_pdb_file, model=1)
                
                # 获取所有唯一的res_name（包括离子和配体）
                unique_resnames = np.unique(ref_atom_array.res_name)
                
                # 处理每个非蛋白质的residue（processed_ligands已在上面定义，用于避免重复）
                for res_name in unique_resnames:
                    res_name_upper = res_name.upper()
                    
                    # 跳过标准氨基酸、DNA、RNA
                    standard_aa = set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                                      'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])
                    if (res_name_upper in standard_aa or
                        res_name_upper in ['DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U', 'HOH', 'WAT']):
                        continue
                    
                    # 获取该residue的所有原子
                    res_atoms = ref_atom_array[ref_atom_array.res_name == res_name]
                    if len(res_atoms) == 0:
                        continue
                    
                    # 跳过已经在backbone中处理过的配体
                    if res_name_upper in processed_ligands:
                        continue
                    
                    # 检查是否是金属离子（单原子）
                    if len(res_atoms) == 1:
                        atom_name = res_atoms.atom_name[0].upper()
                        # 检查是否是已知的金属离子
                        if res_name_upper in ccd_parser.METAL_IONS:
                            smiles = ccd_parser.METAL_IONS[res_name_upper]
                            # 验证SMILES
                            test_mol = Chem.MolFromSmiles(smiles)
                            if test_mol is not None:
                                fasta_strings.append(f">ligand|name={res_name_upper}\n{smiles}\n")
                                processed_ligands.add(res_name_upper)
                                print(f"Added ion {res_name_upper} with SMILES {smiles}")
                            else:
                                print(f"Warning: Ion SMILES '{smiles}' for {res_name_upper} cannot be parsed by RDKit")
                        else:
                            # 尝试从CCD获取
                            smiles_result = ccd_parser.get_smiles(res_name_upper)
                            if smiles_result is not None:
                                if isinstance(smiles_result, list):
                                    smiles = smiles_result[0]
                                else:
                                    smiles = smiles_result
                                smiles = smiles.strip('"\'')
                                test_mol = Chem.MolFromSmiles(smiles)
                                if test_mol is not None:
                                    fasta_strings.append(f">ligand|name={res_name_upper}\n{smiles}\n")
                                    processed_ligands.add(res_name_upper)
                                    print(f"Added ligand {res_name_upper} with SMILES {smiles}")
                    else:
                        # 多原子配体，从CCD获取SMILES
                        if res_name_upper not in processed_ligands:
                            smiles_result = ccd_parser.get_smiles(res_name_upper)
                            if smiles_result is not None:
                                if isinstance(smiles_result, list):
                                    smiles = smiles_result[0]
                                else:
                                    smiles = smiles_result
                                smiles = smiles.strip('"\'')
                                test_mol = Chem.MolFromSmiles(smiles)
                                if test_mol is not None:
                                    fasta_strings.append(f">ligand|name={res_name_upper}\n{smiles}\n")
                                    processed_ligands.add(res_name_upper)
                                    print(f"Added ligand {res_name_upper} with SMILES {smiles}")
                            else:
                                print(f"Warning: Failed to get SMILES for ligand {res_name_upper}")
            except Exception as e:
                print(f"Warning: Failed to read reference PDB file {reference_pdb_path}: {e}")
        
        with open(output_path, 'w') as f:
            f.writelines(fasta_strings)
    
    def make_chai1_fasta_multi_process(self, backbone_dir: str, output_dir: str, origin_cwd: str, inverse_fold_dir: str = None, ame_csv: str = None):
        """
        从 inversefold 输出中提取序列，并使用 assets/ame 中的标准非蛋白 FASTA 生成 chai1 input
        
        Args:
            backbone_dir: backbone PDB 文件目录
            output_dir: 输出 FASTA 文件目录
            origin_cwd: 原始工作目录
            inverse_fold_dir: inversefold 输出目录（包含 seqs/ 子目录），如果为 None 则从 config 推断
            ame_csv: Optional path to AME CSV file (for task name matching)
        """
        import re
        
        backbone_path_list = list(Path(backbone_dir).glob("*.pdb"))
        
        # 确定 inversefold 输出目录
        if inverse_fold_dir is None:
            # 尝试从 config 或默认路径推断
            inverse_fold_dir = getattr(self.config, 'inverse_fold_dir', None)
            if inverse_fold_dir is None:
                # 默认路径：backbone_dir 的父目录的 inverse_fold 子目录
                inverse_fold_dir = str(Path(backbone_dir).parent.parent / "inverse_fold")
        
        inverse_fold_path = Path(inverse_fold_dir)
        seqs_dir = inverse_fold_path / "seqs"
        
        # 查找标准非蛋白 FASTA 目录
        standard_nonprotein_fasta_dir = None
        possible_fasta_dirs = [
            Path(origin_cwd) / "assets" / "ame" / "standard_nonprotein_fasta",
            Path(origin_cwd) / "ODesignBench" / "assets" / "ame" / "standard_nonprotein_fasta",
            Path(__file__).resolve().parents[1] / "assets" / "ame" / "standard_nonprotein_fasta",
        ]
        for fasta_dir in possible_fasta_dirs:
            if fasta_dir.exists():
                standard_nonprotein_fasta_dir = fasta_dir
                break
        
        if standard_nonprotein_fasta_dir is None:
            raise FileNotFoundError(
                f"Standard nonprotein FASTA directory not found. Tried: {possible_fasta_dirs}"
            )
        
        # Load AME CSV if provided for task name matching
        ame_csv_df = None
        if ame_csv is not None and Path(ame_csv).exists():
            from inversefold.ame_csv_utils import load_ame_csv, match_pdb_to_csv_info
            ame_csv_df = load_ame_csv(ame_csv)
            print(f"Loaded AME CSV with {len(ame_csv_df)} entries for task name matching")
        
        def get_task_name_from_backbone(backbone_path: Path) -> str:
            """从 backbone 文件名获取任务名称，优先使用 CSV，否则从文件名提取"""
            # First try to match from CSV if available
            if ame_csv_df is not None:
                csv_info = match_pdb_to_csv_info(backbone_path, ame_csv_df)
                if csv_info is not None:
                    task_name = csv_info.get('task', None)
                    if task_name:
                        return task_name
            
            # Fallback: extract from filename
            filename = backbone_path.stem
            match = re.search(r'([mM]\d{4}_[^_]+)', filename)
            if match:
                task_name_raw = match.group(1)
                return 'M' + task_name_raw[1:]  # 转换为大写 M
            return None
        
        def get_sequence_from_inversefold(backbone_stem: str, seqs_dir: Path) -> str:
            """从 inversefold 输出中提取序列"""
            # 尝试提取 seed index（如果有）
            seed_idx = None
            if "-" in backbone_stem:
                tail = backbone_stem.rsplit("-", 1)[1]
                if tail.isdigit():
                    seed_idx = int(tail)
            
            def _read_fasta_for_seed(fasta_file: Path, seed_index: int | None) -> str:
                """读取 FASTA 文件中的序列"""
                try:
                    with open(fasta_file, "r") as f:
                        lines = [l.rstrip("\n") for l in f]
                except Exception:
                    return ""
                
                sequences = []
                current_header = None
                current_seq = None
                
                for line in lines:
                    if line.startswith(">"):
                        if current_header is not None and current_seq is not None:
                            sequences.append((current_header, current_seq))
                        current_header = line[1:].strip()
                        current_seq = ""
                    else:
                        if current_header is not None:
                            current_seq += line.strip()
                
                if current_header is not None and current_seq is not None:
                    sequences.append((current_header, current_seq))
                
                if not sequences:
                    return ""
                
                # 如果没有指定 seed，返回第一个序列
                if seed_index is None:
                    seq = sequences[0][1]
                    seq = seq.replace(":", "").replace(";", "")
                    return seq.upper()
                
                # 尝试通过 header 中的 id=N 匹配
                for header, seq in sequences:
                    m = re.search(r"id=(\d+)", header)
                    if m and int(m.group(1)) == seed_index:
                        seq = seq.replace(":", "").replace(";", "")
                        return seq.upper()
                
                # 回退：使用第 N 个序列（1-based）
                idx = seed_index - 1
                if 0 <= idx < len(sequences):
                    seq = sequences[idx][1]
                    seq = seq.replace(":", "").replace(";", "")
                    return seq.upper()
                
                # 最终回退：第一个序列
                seq = sequences[0][1]
                seq = seq.replace(":", "").replace(";", "")
                return seq.upper()
            
            if not seqs_dir.exists():
                return None
            
            # 尝试精确匹配 sample_name
            for fasta_file in seqs_dir.glob(f"{backbone_stem}*.fa"):
                seq = _read_fasta_for_seed(fasta_file, seed_idx)
                if seq:
                    return seq
            
            # 回退：设计级别的 .fa 文件（包含多个序列）
            design_name = backbone_stem.rsplit("-", 1)[0] if "-" in backbone_stem else backbone_stem
            if design_name:
                for fasta_file in seqs_dir.glob(f"{design_name}*.fa"):
                    seq = _read_fasta_for_seed(fasta_file, seed_idx)
                    if seq:
                        return seq
            
            return None
        
        def read_standard_nonprotein_fasta(nonprotein_fasta_path: Path) -> list:
            """读取标准非蛋白 entity FASTA 文件"""
            if not nonprotein_fasta_path.exists():
                return []
            
            entries = []
            current_header = None
            current_sequence = []
            
            with open(nonprotein_fasta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        if current_header is not None and current_sequence:
                            entries.append((current_header, ''.join(current_sequence)))
                        current_header = line
                        current_sequence = []
                    else:
                        current_sequence.append(line)
                
                if current_header is not None and current_sequence:
                    entries.append((current_header, ''.join(current_sequence)))
            
            return entries
        
        def generate_chai1_input_fasta(backbone_path: Path, output_path: Path) -> bool:
            """生成 chai1 input FASTA 文件"""
            backbone_stem = backbone_path.stem
            
            # 1. 从 inversefold 输出中提取蛋白质序列
            protein_sequence = get_sequence_from_inversefold(backbone_stem, seqs_dir)
            if protein_sequence is None:
                print(f"Warning: Failed to extract sequence from inversefold output for {backbone_stem}")
                return False
            
            # 2. 从 backbone 文件名获取任务名称（优先使用 CSV）
            task_name = get_task_name_from_backbone(backbone_path)
            if task_name is None:
                print(f"Warning: Failed to get task name from {backbone_stem}")
                return False
            
            # 3. 读取对应的标准非蛋白 entity FASTA 文件
            nonprotein_fasta_path = standard_nonprotein_fasta_dir / f"{task_name}_nonprotein.fasta"
            nonprotein_entries = read_standard_nonprotein_fasta(nonprotein_fasta_path)
            
            if not nonprotein_entries:
                print(f"Warning: No nonprotein entries found in {nonprotein_fasta_path}")
            
            # 4. 合并生成完整的 chai1 input FASTA 文件
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                # 写入蛋白质序列
                f.write(f">protein|name=protein\n{protein_sequence}\n")
                
                # 写入所有非蛋白 entity（配体和离子）
                for header, sequence in nonprotein_entries:
                    f.write(f"{header}\n{sequence}\n")
            
            return True
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.refold.num_workers) as executor:
            futures = []
            for backbone_path in backbone_path_list:
                output_path = Path(output_dir) / f"{backbone_path.stem}.fasta"
                future = executor.submit(generate_chai1_input_fasta, backbone_path, output_path)
                futures.append(future)
            
            success_count = 0
            fail_count = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"Error processing file: {e}")
                    fail_count += 1
            
            print(f"\n完成！成功: {success_count}, 失败: {fail_count}")
            print(f"输出目录: {output_dir}")
        return self._make_result(
            stage="refold.make_chai1_fasta_multi_process",
            outputs={"output_dir": str(output_dir)},
            details={
                "num_backbones": len(backbone_path_list),
                "success_count": success_count,
                "fail_count": fail_count,
            },
        )
