import os
import sys
import tqdm
import pickle
import operator
import subprocess
import shutil
from time import perf_counter
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import concurrent.futures
from evaluation.metrics.rmsd import RMSDCalculator
from evaluation.metrics.confidence import Confidence
from evaluation.metrics.usalign import USalign
from evaluation.metrics.foldseek import FoldSeek
import json
import glob
import re
from typing import Optional, Any

class Evaluation():

    def __init__(self, config):

        self.config = config

    def _normalize_result(self, task: str, raw_result: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        details: dict[str, Any] = {}
        success = True

        output_csv = kwargs.get("output_csv")
        output_dir = kwargs.get("output_dir")
        if output_csv is not None:
            outputs["output_csv"] = str(output_csv)
            success = success and os.path.exists(str(output_csv))
        if output_dir is not None:
            outputs["output_dir"] = str(output_dir)

        if isinstance(raw_result, dict):
            details["raw_result"] = raw_result
            success = success and bool(raw_result.get("success", True))
        elif hasattr(raw_result, "to_csv"):
            details["return_type"] = "dataframe"
        elif raw_result is not None:
            details["return_type"] = type(raw_result).__name__

        return {
            "success": bool(success),
            "stage": f"evaluation.{task}",
            "task": task,
            "outputs": outputs,
            "details": details,
        }

    def run(self, task: str, **kwargs) -> dict[str, Any]:
        dispatch = {
            "protein": self.run_protein_evaluation,
            "pbp": self.run_protein_binding_protein_evaluation,
            "lbp": self.run_ligand_binding_protein_evaluation,
            "nuc": self.run_nuc_evaluation,
            "nbl": self.run_nuc_binding_ligand_evaluation,
            "pbn": self.run_protein_binding_nuc_evaluation,
            "pbl": self.run_protein_binding_ligand_evaluation,
            "ame": self.run_atomic_motif_enzyme_evaluation,
            "motif_scaffolding": self.run_motif_scaffolding_evaluation,
        }
        if task not in dispatch:
            known = ", ".join(sorted(dispatch.keys()))
            raise ValueError(f"Unknown evaluation task '{task}'. Supported: {known}")
        t0 = perf_counter()
        raw = dispatch[task](**kwargs)
        result = self._normalize_result(task=task, raw_result=raw, kwargs=kwargs)
        elapsed = perf_counter() - t0
        details = result.setdefault("details", {})
        details["elapsed_seconds"] = round(elapsed, 3)
        print(f"[timing] evaluation.{task}: {elapsed:.2f}s")
        return result

    def _compute_foldseek_diversity_and_novelty(
        self,
        pipeline_dir: str,
        df: pd.DataFrame,
        output_csv: str,
        subdir_suffix: str,
        include_designability: bool = True,
    ) -> pd.DataFrame:
        """
        Helper to compute FoldSeek-based diversity and novelty for a set of designs.
        Mirrors the logic used in run_protein_evaluation and motif scaffolding.
        """
        # Derive design names from DataFrame index.
        # Some tasks produce multiple refold rows per design (e.g., xxx-1 ... xxx-8),
        # while formatted_designs keeps one structure per base design.
        design_names = list(df.index)
        total_designs = len(design_names)

        formatted_designs_dir = os.path.join(pipeline_dir, "formatted_designs")
        resolved_design_names: list[str] = []
        missing_design_names: list[str] = []
        seen_resolved: set[str] = set()
        for design_name in design_names:
            candidates = [design_name]
            if "-" in design_name:
                base_name, tail = design_name.rsplit("-", 1)
                if tail.isdigit():
                    candidates.append(base_name)

            resolved_name = None
            for cand in candidates:
                if os.path.exists(os.path.join(formatted_designs_dir, f"{cand}.pdb")):
                    resolved_name = cand
                    break

            if resolved_name is None:
                missing_design_names.append(design_name)
                continue

            # Deduplicate so one formatted structure corresponds to many refold outputs.
            if resolved_name not in seen_resolved:
                seen_resolved.add(resolved_name)
                resolved_design_names.append(resolved_name)

        num_designable = len(resolved_design_names)
        designability = (num_designable / total_designs) if total_designs > 0 else 0.0

        # Initialize summary data
        summary_data = {
            'diversity': {
                'value': 0.0,
                'clusters': 0,
                'num_designable': num_designable,
            },
            'novelty': {
                'value': 0.0,
                'max_tmscore_avg': 0.0,
                'num_designable': num_designable,
            },
        }
        if include_designability:
            summary_data['designability'] = {
                'value': designability,
                'num_designable': num_designable,
                'total_designs': total_designs,
            }

        if total_designs == 0 or num_designable == 0:
            if total_designs == 0:
                print("Warning: No designs found for FoldSeek diversity/novelty.")
            else:
                print(
                    "Warning: No mappable formatted designs found for FoldSeek diversity/novelty."
                )
            df['diversity'] = 0.0
            df['novelty'] = 0.0
            if include_designability:
                df['designability'] = designability
        else:
            # Prepare structure directory using formatted_designs (one PDB per design)
            metrics_output_dir = os.path.join(pipeline_dir, "metrics")
            designable_structure_dir = os.path.join(
                metrics_output_dir, f"designable_structures_{subdir_suffix}"
            )
            os.makedirs(designable_structure_dir, exist_ok=True)

            # Copy designable structure files from formatted_designs to temporary directory
            for design_name in resolved_design_names:
                src_path = os.path.join(formatted_designs_dir, f"{design_name}.pdb")
                dst_path = os.path.join(designable_structure_dir, f"{design_name}.pdb")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: Design structure not found for FoldSeek: {src_path}")
            if missing_design_names:
                print(
                    "Warning: Some evaluation rows could not be mapped to formatted_designs. "
                    f"Missing count: {len(missing_design_names)}"
                )

            print(
                f"Computing FoldSeek diversity and novelty for {len(resolved_design_names)} designs "
                f"(task={subdir_suffix})..."
            )

            # Initialize FoldSeek with config
            foldseek_config = {}
            # Prefer metrics-level config if available
            if hasattr(self.config, 'metrics'):
                metrics_cfg = self.config.metrics
                if hasattr(metrics_cfg, 'foldseek_bin'):
                    foldseek_config['foldseek_bin'] = metrics_cfg.foldseek_bin
                if hasattr(metrics_cfg, 'foldseek_database'):
                    foldseek_config['foldseek_database'] = metrics_cfg.foldseek_database
                foldseek_config['verbose'] = getattr(metrics_cfg, 'verbose', False)

            # Fallback: try to load from foldseek.yaml if database not set
            if 'foldseek_database' not in foldseek_config or not foldseek_config['foldseek_database']:
                import yaml
                foldseek_yaml_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), 'configs', 'foldseek.yaml'
                )
                if os.path.exists(foldseek_yaml_path):
                    try:
                        with open(foldseek_yaml_path, 'r') as f:
                            foldseek_yaml = yaml.safe_load(f)
                            if 'foldseek_bin' in foldseek_yaml and 'foldseek_bin' not in foldseek_config:
                                foldseek_config['foldseek_bin'] = foldseek_yaml['foldseek_bin']
                            if 'foldseek_database' in foldseek_yaml:
                                foldseek_config['foldseek_database'] = foldseek_yaml['foldseek_database']
                            print(f"Loaded FoldSeek config from {foldseek_yaml_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load foldseek.yaml: {e}")

            print(
                f"FoldSeek config - bin: {foldseek_config.get('foldseek_bin', 'NOT SET')}"
            )
            print(
                f"FoldSeek config - database: {foldseek_config.get('foldseek_database', 'NOT SET')}"
            )

            foldseek = FoldSeek(foldseek_config)

            # Compute diversity
            diversity_result = foldseek.compute_diversity(
                structure_dir=designable_structure_dir,
                output_dir=metrics_output_dir,
                dump=True,
            )

            # Compute novelty
            novelty_result = foldseek.compute_novelty(
                structure_dir=designable_structure_dir,
                output_dir=metrics_output_dir,
                dump=True,
            )

            # Diversity value: clusters / num_designable
            num_clusters = diversity_result.get('num_clusters', 0)
            diversity_value = num_clusters / num_designable if num_designable > 0 else 0.0
            summary_data['diversity'] = {
                'value': diversity_value,
                'clusters': num_clusters,
                'num_designable': num_designable,
            }

            # Novelty summary
            novelty_value = novelty_result.get('novelty', 0.0)
            summary_data['novelty'] = {
                'value': novelty_value,
                'max_tmscore_avg': novelty_result.get('max_tmscore_avg', 0.0),
                'num_designable': num_designable,
            }

            # Attach per-sample diversity/novelty (and optional designability) (same for all rows)
            df['diversity'] = diversity_result.get('diversity', 0.0)
            df['novelty'] = novelty_result.get('novelty', 0.0)
            if include_designability:
                df['designability'] = designability

        # Save summary JSON next to output CSV
        output_dir = os.path.dirname(output_csv)
        summary_output_path = os.path.join(output_dir, f'raw_summary_{subdir_suffix}.json')
        with open(summary_output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary (FoldSeek diversity/novelty) saved to {summary_output_path}")

        return df
    
    @staticmethod
    def _get_sequence_from_inversefold(pipeline_dir: str, sample_name: str) -> str:
        """
        Get sequence from LigandMPNN/ProteinMPNN output fasta file

        Args:
            pipeline_dir: pipeline root directory
            sample_name: sample name (without extension)
            
        Returns:
            str: protein sequence, if not found return empty string
        """
        import re

        # Optional seed index: sample names like "01_7UXQ_0-7" -> seed_idx=7
        seed_idx = None
        if "-" in sample_name:
            tail = sample_name.rsplit("-", 1)[1]
            if tail.isdigit():
                seed_idx = int(tail)

        def _read_fasta_for_seed(fasta_file: Path, seed_index: int | None) -> str:
            """
            Read a multi-fasta from LigandMPNN.
            - If seed_index is given, try to return the sequence whose header has matching id=N
              (e.g. \">01_7UXQ_0, id=7, ...\").
            - If no explicit id is found, fall back to N-th sequence in the file (1-based).
            - If seed_index is None, return the first sequence.
            """
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
                    # flush previous
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

            # If no specific seed requested, return first sequence
            if seed_index is None:
                seq = sequences[0][1]
                seq = seq.replace(":", "").replace(";", "")
                return seq.upper()

            # Try to match by explicit "id=N" in header
            for header, seq in sequences:
                m = re.search(r"id=(\\d+)", header)
                if m and int(m.group(1)) == seed_index:
                    seq = seq.replace(":", "").replace(";", "")
                    return seq.upper()

            # Fallback: use N-th sequence if available (1-based)
            idx = seed_index - 1
            if 0 <= idx < len(sequences):
                seq = sequences[idx][1]
                seq = seq.replace(":", "").replace(";", "")
                return seq.upper()

            # Final fallback: first sequence
            seq = sequences[0][1]
            seq = seq.replace(":", "").replace(";", "")
            return seq.upper()

        seqs_dir = os.path.join(pipeline_dir, "inverse_fold", "seqs")
        if not os.path.exists(seqs_dir):
            return ""

        # Try exact sample_name first (rare; usually we only have design-level .fa)
        for fasta_file in Path(seqs_dir).glob(f"{sample_name}*.fa"):
            seq = _read_fasta_for_seed(fasta_file, seed_idx)
            if seq:
                return seq

        # Fallback: design-level .fa (e.g. 01_7UXQ_0.fa) that contains multiple sequences (seeds)
        design_name = sample_name.rsplit("-", 1)[0] if "-" in sample_name else sample_name
        if design_name:
            for fasta_file in Path(seqs_dir).glob(f"{design_name}*.fa"):
                seq = _read_fasta_for_seed(fasta_file, seed_idx)
                if seq:
                    return seq

        return ""
    

    def run_protein_binding_ligand_evaluation(self, input_dir: str, output_dir: str, 
                          dist_cutoff: float = 10.0, 
                          exhaustiveness: int = 16,
                          num_processes: int = 8,
                          verbose: bool = True,
                          cuda_device: str = "0",
                          enable_geom: bool = True,
                          enable_chem: bool = True,
                          enable_vina: bool = True,
                          ccd_bond_length_path: str = None,
                          ccd_bond_angle_path: str = None,
                          ccd_torsion_angle_path: str = None):
        """
        Run ligand evaluation pipeline using the ligand_evaluation.py script
        
        Args:
            input_dir (str): Input directory containing CIF files
            output_dir (str): Output directory for results
            dist_cutoff (float): Distance cutoff for pocket definition (default: 10.0)
            exhaustiveness (int): Exhaustiveness for docking (default: 16)
            num_processes (int): Number of processes (default: 8)
            verbose (bool): Enable verbose output (default: True)
            cuda_device (str): CUDA device to use (default: "0")
            enable_geom (bool): Enable geometry evaluation (default: True)
            enable_chem (bool): Enable chemistry evaluation (default: True)
            enable_vina (bool): Enable Vina docking evaluation (default: True)
            ccd_bond_length_path (str): Path to CCD bond length distribution file
            ccd_bond_angle_path (str): Path to CCD bond angle distribution file
            ccd_torsion_angle_path (str): Path to CCD torsion angle distribution file
            
        Returns:
            dict: Result dictionary containing success status and output information
        """
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = cuda_device

            executable_path = Path(sys.executable).resolve()
            env_prefix = executable_path.parent.parent
            env_bin = str(env_prefix / "bin")
            env_lib = str(env_prefix / "lib")
            babel_libdir = str(getattr(self.config.metrics, "babel_libdir", "") or "").strip()
            if not babel_libdir:
                autodetect_babel_libdir = env_prefix / "lib" / "openbabel" / "3.1.0"
                if autodetect_babel_libdir.exists():
                    babel_libdir = str(autodetect_babel_libdir)
            if babel_libdir:
                env["BABEL_LIBDIR"] = babel_libdir

            babel_datadir = env_prefix / "share" / "openbabel" / "3.1.0"
            if babel_datadir.exists():
                env["BABEL_DATADIR"] = str(babel_datadir)

            env["PATH"] = (
                f"{env_bin}:{env['PATH']}" if env.get("PATH") else env_bin
            )
            env["LD_LIBRARY_PATH"] = (
                f"{env_lib}:{env['LD_LIBRARY_PATH']}"
                if env.get("LD_LIBRARY_PATH")
                else env_lib
            )

            
            # Construct the command
            script_path = os.path.join(os.path.dirname(__file__), "ligand_evaluation.py")
            
            cmd = [
                sys.executable, script_path,
                '--input_dir', input_dir,
                '--output_dir', output_dir,
                '--dist_cutoff', str(dist_cutoff),
                '--exhaustiveness', str(exhaustiveness),
                '--num_processes', str(num_processes),
                '--ccd_bond_length_path', self.config.metrics.ccd_bond_length_path,
                '--ccd_bond_angle_path', self.config.metrics.ccd_bond_angle_path,
                '--ccd_torsion_angle_path', self.config.metrics.ccd_torsion_angle_path
            ]
            
            # Add evaluation module flags
            if enable_geom:
                cmd.append('--enable_geom')
            else:
                cmd.append('--disable_geom')
                
            if enable_chem:
                cmd.append('--enable_chem')
            else:
                cmd.append('--disable_chem')
                
            if enable_vina:
                cmd.append('--enable_vina')
            else:
                cmd.append('--disable_vina')
            
            # Add verbose flag if requested
            if verbose:
                cmd.append('--verbose')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run the command
            print(f"Running ligand metrics pipeline...")
            print(f"Command: {' '.join(cmd)}")
            print(f"Input directory: {input_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Input directory exists: {os.path.exists(input_dir)}")
            if os.path.exists(input_dir):
                print(f"Files in input directory: {os.listdir(input_dir)}")
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                # cwd=current_dir
            )
            
            # Prepare result dictionary
            result_dict = {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'input_dir': input_dir,
                'output_dir': output_dir,
                'command': ' '.join(cmd)
            }
            
            if result.returncode == 0:
                print("Ligand metrics pipeline completed successfully!")
                print(f"Results saved to: {output_dir}")
            else:
                print(f"Ligand metrics pipeline failed with return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
            
            return result_dict
            
        except Exception as e:
            error_dict = {
                'success': False,
                'error': str(e),
                'input_dir': input_dir,
                'output_dir': output_dir
            }
            print(f"Error running ligand metrics pipeline: {e}")
            return error_dict
    
    def run_protein_evaluation(self, pipeline_dir: str, output_csv: str, ca_rmsd_threshold: float = 2.0):
        '''
        refold model: esmfold
        '''
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", sample_name)

                ca_rmsd = RMSDCalculator.compute_protein_ca_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                plddt = Confidence.gather_esmfold_confidence(str(refold_path))
                result_data = {
                    'ca_rmsd': ca_rmsd,
                    'plddt': plddt,
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: get error when dealing with {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)

        esmfold_result_json_path = os.path.join(pipeline_dir, "refold", "esmfold_out", "esmfold_results.json")
        with open(esmfold_result_json_path, "r") as f:
            esmfold_result_json = json.load(f)

        all_refold_paths = [Path(i['pdb_path']) for i in esmfold_result_json]
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        
        # Extract design names from sample names (e.g., oqo-1-4.pdb -> oqo-1)
        def extract_design_name(sample_name: str) -> str:
            # Remove .pdb extension and extract design name (remove last number)
            name_without_ext = sample_name.replace('.pdb', '')
            # Split by '-' and remove the last part (refold number)
            parts = name_without_ext.rsplit('-', 1)
            if len(parts) == 2:
                return parts[0]
            return name_without_ext
        
        # Add design_name column
        df['design_name'] = df.index.map(extract_design_name)
        
        # Mark structures as designable if ca_rmsd < threshold
        df['is_designable'] = df['ca_rmsd'] < ca_rmsd_threshold
        
        # Group by design_name and check if all refold structures are designable
        design_groups = df.groupby('design_name')
        designable_designs = set()
        for design_name, group in design_groups:
            # A design is designable if all its refold structures have ca_rmsd < threshold
            if group['is_designable'].all():
                designable_designs.add(design_name)
        
        # Calculate designability
        total_designs = len(design_groups)
        num_designable = len(designable_designs)
        designability = num_designable / total_designs if total_designs > 0 else 0.0
        
        print(f"Designability: {num_designable}/{total_designs} = {designability:.4f}")
        
        # Filter to only designable structures for diversity and novelty computation
        designable_df = df[df['design_name'].isin(designable_designs)]
        
        # Initialize summary data
        summary_data = {
            'designability': {
                'value': designability,
                'num_designable': num_designable,
                'total_designs': total_designs
            },
            'diversity': {
                'value': 0.0,
                'clusters': 0,
                'num_designable': num_designable
            },
            'novelty': {
                'value': 0.0,
                'max_tmscore_avg': 0.0,
                'num_designable': num_designable
            }
        }
        
        if len(designable_df) == 0:
            print("Warning: No designable structures found. Setting diversity and novelty to 0.0")
            df['diversity'] = 0.0
            df['novelty'] = 0.0
            df['designability'] = designability
        else:
            # Create a temporary directory with only designable structures
            # Use formatted_designs instead of refold results to avoid duplicates
            # (refold generates multiple structures per design based on MPNN count)
            formatted_designs_dir = os.path.join(pipeline_dir, "formatted_designs")
            metrics_output_dir = os.path.join(pipeline_dir, "metrics")
            designable_structure_dir = os.path.join(metrics_output_dir, "designable_structures")
            os.makedirs(designable_structure_dir, exist_ok=True)
            
            # Copy designable structure files from formatted_designs to temporary directory
            # Use design names (e.g., oqo-1) instead of refold sample names (e.g., oqo-1-1.pdb)
            for design_name in designable_designs:
                src_path = os.path.join(formatted_designs_dir, f"{design_name}.pdb")
                dst_path = os.path.join(designable_structure_dir, f"{design_name}.pdb")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: Design structure not found: {src_path}")
            
            print(f"Computing diversity and novelty for {len(designable_designs)} designable structures...")
            
            # Initialize FoldSeek with config
            foldseek_config = {}
            if hasattr(self.config.metrics, 'foldseek_bin'):
                foldseek_config['foldseek_bin'] = self.config.metrics.foldseek_bin
            if hasattr(self.config.metrics, 'foldseek_database'):
                foldseek_config['foldseek_database'] = self.config.metrics.foldseek_database
            foldseek_config['verbose'] = getattr(self.config.metrics, 'verbose', False)
            
            # Fallback: try to load from foldseek.yaml if not in metrics config
            if 'foldseek_database' not in foldseek_config or not foldseek_config['foldseek_database']:
                import yaml
                foldseek_yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'foldseek.yaml')
                if os.path.exists(foldseek_yaml_path):
                    try:
                        with open(foldseek_yaml_path, 'r') as f:
                            foldseek_yaml = yaml.safe_load(f)
                            if 'foldseek_bin' in foldseek_yaml and 'foldseek_bin' not in foldseek_config:
                                foldseek_config['foldseek_bin'] = foldseek_yaml['foldseek_bin']
                            if 'foldseek_database' in foldseek_yaml:
                                foldseek_config['foldseek_database'] = foldseek_yaml['foldseek_database']
                            print(f"Loaded FoldSeek config from {foldseek_yaml_path}")
                    except Exception as e:
                        print(f"Warning: Failed to load foldseek.yaml: {e}")
            
            # Debug: print configuration
            print(f"FoldSeek config - bin: {foldseek_config.get('foldseek_bin', 'NOT SET')}")
            print(f"FoldSeek config - database: {foldseek_config.get('foldseek_database', 'NOT SET')}")
            
            foldseek = FoldSeek(foldseek_config)
            
            # Compute diversity for designable structures
            diversity_result = foldseek.compute_diversity(
                structure_dir=designable_structure_dir,
                output_dir=metrics_output_dir,
                dump=True
            )
            
            # Compute novelty for designable structures
            novelty_result = foldseek.compute_novelty(
                structure_dir=designable_structure_dir,
                output_dir=metrics_output_dir,
                dump=True
            )
            
            # Update summary data with diversity results
            # Diversity should be based on designable_designs: num_clusters / num_designable
            num_clusters = diversity_result.get('num_clusters', 0)
            diversity_value = num_clusters / num_designable if num_designable > 0 else 0.0
            summary_data['diversity'] = {
                'value': diversity_value,
                'clusters': num_clusters,
                'num_designable': num_designable
            }
            
            # Update summary data with novelty results
            # Novelty value is already computed
            novelty_value = novelty_result.get('novelty', 0.0)
            summary_data['novelty'] = {
                'value': novelty_value,
                'max_tmscore_avg': novelty_result.get('max_tmscore_avg', 0.0),
                'num_designable': num_designable
            }
            
            # Add diversity and novelty to all rows (same value for all samples)
            df['diversity'] = diversity_result.get('diversity', 0.0)
            df['novelty'] = novelty_result.get('novelty', 0.0)
            df['designability'] = designability
        
        # Save summary JSON
        output_dir = os.path.dirname(output_csv)
        summary_output_path = os.path.join(output_dir, 'raw_summary.json')
        with open(summary_output_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary saved to {summary_output_path}")
        
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")

    # maybe need to seperate into different functions for easy use and development
    def run_nuc_binding_ligand_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", f"{sample_name.upper()}.cif")
                trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}-1.pkl")
                summary_confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_summary_confidences.json")
                confidence_path = os.path.join(refold_path.parent, f"{refold_path.parent.name}_confidences.json")

                # breakpoint()
                
                
                plddt, ipae, min_ipae, iptm, ptm_binder, _ptm_H, _ptm_L = Confidence.gather_af3_confidence(
                    confidence_path, summary_confidence_path, trb_path
                )
                # breakpoint()
                ligand_rmsd = RMSDCalculator.compute_nuc_align_ligand_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                result_data = {
                    'ligand_rmsd': ligand_rmsd,
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")

    # maybe need to seperate into different functions for easy use and development
    def run_protein_binding_protein_evaluation(
        self,
        pipeline_dir: str,
        output_csv: str,
        cdr_info_csv: Optional[str] = None,
        pbp_info_csv: Optional[str] = None,
    ):
        
        cdr_df = None
        if cdr_info_csv is not None and os.path.exists(cdr_info_csv):
            try:
                from inversefold.cdr_utils import match_pdb_to_cdr_info
                cdr_df = pd.read_csv(cdr_info_csv)
            except Exception:
                cdr_df = None

        pbp_info_map = None
        if pbp_info_csv is not None:
            if not os.path.exists(pbp_info_csv):
                raise FileNotFoundError(f"PBP info CSV not found: {pbp_info_csv}")
            from inversefold.pbp_csv_utils import load_pbp_info_csv

            pbp_df = load_pbp_info_csv(pbp_info_csv)
            pbp_info_map = {}
            for _, row in pbp_df.iterrows():
                key = Path(str(row["design_name"])).stem
                value = {
                    "target_chain": str(row["target_chain"]).strip(),
                    "design_chain": str(row["design_chain"]).strip(),
                }
                # Keep both original and lowercase keys so AF3 outputs with
                # case-normalized filenames can resolve metadata robustly.
                pbp_info_map[key] = value
                pbp_info_map[key.lower()] = value

        def _parse_chain_csv(val):
            if val is None:
                return []
            s = str(val).strip()
            if not s or s.lower() == "nan":
                return []
            return [p.strip() for p in s.split(",") if p.strip()]

        def _normalize_af3_sample_name(name: str) -> str:
            sample_name = str(name).strip().replace("_model", "")
            sample_name = re.sub(r"_seed-\d+_sample-\d+$", "", sample_name)
            return sample_name

        def _resolve_confidence_path(refold_path: Path, sample_name: str, suffix: str) -> str:
            parents = [refold_path.parent]
            if refold_path.parent.name.startswith("seed-") and refold_path.parent.parent != refold_path.parent:
                parents.append(refold_path.parent.parent)
            for parent in parents:
                candidate = parent / f"{sample_name}{suffix}"
                if candidate.exists():
                    return str(candidate)
            return str(parents[0] / f"{sample_name}{suffix}")

        def _collect_latest_af3_outputs(root_dir: Path) -> tuple[list[Path], dict[str, int]]:
            all_refold_paths = list(root_dir.rglob("*_model.cif"))
            sample_to_path: dict[str, Path] = {}
            duplicate_counts: dict[str, int] = defaultdict(int)
            for p in all_refold_paths:
                sample_name = _normalize_af3_sample_name(p.stem)
                if sample_name in sample_to_path:
                    duplicate_counts[sample_name] += 1
                    chosen = sample_to_path[sample_name]
                    chosen_depth = len(chosen.relative_to(root_dir).parts)
                    cand_depth = len(p.relative_to(root_dir).parts)
                    if cand_depth < chosen_depth or (cand_depth == chosen_depth and p.stat().st_mtime > chosen.stat().st_mtime):
                        sample_to_path[sample_name] = p
                else:
                    sample_to_path[sample_name] = p
            return list(sample_to_path.values()), duplicate_counts

        complex_refold_root = Path(os.path.join(pipeline_dir, "refold", "af3_out"))
        unbound_refold_root = Path(os.path.join(pipeline_dir, "refold", "af3_unbound_out"))
        backbones_dir = Path(pipeline_dir) / "inverse_fold" / "backbones"
        backbone_name_map: dict[str, Path] = {
            p.name.lower(): p for p in backbones_dir.glob("*.pdb")
        }
        unbound_refold_map: dict[str, Path] = {}
        if unbound_refold_root.exists():
            unbound_refold_paths, unbound_duplicate_counts = _collect_latest_af3_outputs(unbound_refold_root)
            unbound_refold_map = {
                _normalize_af3_sample_name(path.stem): path
                for path in unbound_refold_paths
            }
            if unbound_duplicate_counts:
                print(
                    f"Detected duplicate unbound AF3 outputs for {len(unbound_duplicate_counts)} samples "
                    f"({sum(unbound_duplicate_counts.values())} extra files). Keeping newest per sample."
                )
        else:
            print(f"Warning: Unbound AF3 output directory not found: {unbound_refold_root}")

        def process_metrics_worker(refold_path: Path):
            try:
                # sample_name from CIF filename (e.g. h3-5-3_model.cif -> h3-5-3), not parent dir,
                # so timestamped folders (h3-5-3_20260301_125913) map correctly to backbone h3-5-3.pdb
                sample_name = _normalize_af3_sample_name(refold_path.stem)
                matched_backbone = backbone_name_map.get(f"{sample_name}.pdb".lower())
                inverse_fold_path = (
                    str(matched_backbone)
                    if matched_backbone is not None
                    else os.path.join(pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb")
                )
                # trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                summary_confidence_path = _resolve_confidence_path(refold_path, sample_name, "_summary_confidences.json")
                confidence_path = _resolve_confidence_path(refold_path, sample_name, "_confidences.json")
                
                target_chain_id = None
                design_chain_id = None
                if pbp_info_map is not None:
                    lookup_keys = [sample_name]
                    if "-" in sample_name:
                        base_name, tail = sample_name.rsplit("-", 1)
                        if tail.isdigit():
                            lookup_keys.append(base_name)
                    expanded_keys = []
                    for k in lookup_keys:
                        expanded_keys.append(k)
                        expanded_keys.append(k.lower())
                    for k in expanded_keys:
                        info = pbp_info_map.get(k)
                        if info is not None:
                            target_chain_id = info.get("target_chain")
                            design_chain_id = info.get("design_chain")
                            break
                    if target_chain_id is None or design_chain_id is None:
                        raise ValueError(
                            f"No PBP chain-role metadata found for sample '{sample_name}'. "
                            f"Tried keys: {expanded_keys}"
                        )

                # Complex RMSD should be computed on the target+design chains.
                try:
                    complex_chain_ids = [
                        c
                        for c in [target_chain_id, design_chain_id]
                        if c and c.lower() != "nan"
                    ]
                    if not complex_chain_ids:
                        complex_chain_ids = ["A", "B"]
                    ca_rmsd_complex = RMSDCalculator.compute_protein_ca_rmsd_chain_subset(
                        pred=str(refold_path),
                        refold=inverse_fold_path,
                        chain_ids=complex_chain_ids,
                    )
                    if np.isnan(ca_rmsd_complex):
                        raise ValueError("insufficient common CA residues on target/design complex chains")
                except Exception:
                    ca_rmsd_complex = np.inf
                    print(f"{refold_path} fail for calculate rmsd, set to inf, please check the case")

                # Design RMSD should only use the design chain from pbp_info.
                try:
                    if design_chain_id is None or design_chain_id.lower() == "nan":
                        raise ValueError("design_chain not found in pbp_info")
                    ca_rmsd = RMSDCalculator.compute_protein_ca_rmsd_chain_subset(
                        pred=str(refold_path),
                        refold=inverse_fold_path,
                        chain_ids=[design_chain_id],
                    )
                    if np.isnan(ca_rmsd):
                        raise ValueError("insufficient common CA residues on design chain")
                except Exception:
                    ca_rmsd = np.inf

                try:
                    if design_chain_id is None or design_chain_id.lower() == "nan":
                        raise ValueError("design_chain not found in pbp_info")
                    unbound_refold_path = unbound_refold_map.get(sample_name)
                    if unbound_refold_path is None:
                        raise FileNotFoundError(
                            f"Unbound AF3 output not found for sample '{sample_name}' under {unbound_refold_root}"
                        )
                    ca_rmsd_bound_unbound = RMSDCalculator.compute_protein_ca_rmsd_unbound_vs_bound_design(
                        unbound=str(unbound_refold_path),
                        bound_complex=str(refold_path),
                        bound_chain_ids=[design_chain_id],
                    )
                    if np.isnan(ca_rmsd_bound_unbound):
                        raise ValueError("insufficient common CA residues between unbound and bound design structures")
                except Exception as e:
                    ca_rmsd_bound_unbound = np.nan
                    print(f"Warning: Failed bound/unbound RMSD for {sample_name}: {e}")

                # Chain IDs (Ab: H+L, Ag: A,B,...) from CDR CSV if available
                h_chain_id = None
                l_chain_id = None
                ag_chain_ids = []
                if cdr_df is not None:
                    try:
                        from inversefold.cdr_utils import match_pdb_to_cdr_info
                        cdr_row = match_pdb_to_cdr_info(Path(inverse_fold_path), cdr_df)
                        if cdr_row is not None:
                            h_chain_id = str(cdr_row.get("h_chain")).strip() if cdr_row.get("h_chain") is not None else None
                            l_chain_id = str(cdr_row.get("l_chain")).strip() if cdr_row.get("l_chain") is not None else None
                            ag_chain_ids = _parse_chain_csv(cdr_row.get("ag_chain"))
                    except Exception:
                        pass

                    ab_chain_ids = [c for c in [h_chain_id, l_chain_id] if c and c.lower() != "nan"]

                (
                    plddt,
                    ipae,
                    min_ipae,
                    iptm,
                    ptm_binder,
                    _ptm_H,
                    _ptm_L,
                ) = Confidence.gather_af3_confidence(
                    confidence_path,
                    summary_confidence_path,
                    inverse_fold_path,
                    h_chain_id=h_chain_id,
                    l_chain_id=l_chain_id,
                    ag_chain_ids=ag_chain_ids,
                )

                result_data = {
                    # RMSD (design-chain and complex views)
                    'ca_rmsd': ca_rmsd,
                    'ca_rmsd_complex': ca_rmsd_complex,
                    'ca_rmsd_bound_unbound': ca_rmsd_bound_unbound,
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                }
                return sample_name, result_data
            
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths, duplicate_counts = _collect_latest_af3_outputs(complex_refold_root)
        print(f"{len(all_refold_paths)} unique files were found for evaluation.")
        if duplicate_counts:
            print(
                f"Detected duplicate AF3 outputs for {len(duplicate_counts)} samples "
                f"({sum(duplicate_counts.values())} extra files). Keeping newest per sample."
            )
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df['success'] = (
            (df.get('ipae', np.nan) < 10.85)
            & (df.get('iptm', np.nan) > 0.5)
            & (df.get('plddt', np.nan) > 0.8)
            & (df.get('ca_rmsd_complex', np.nan) < 2.5)
        )
        kept_columns = ['ca_rmsd', 'ca_rmsd_complex', 'ca_rmsd_bound_unbound', 'plddt', 'ipae', 'min_ipae', 'iptm', 'success']
        df = df[[c for c in kept_columns if c in df.columns]]
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")

    # maybe need to seperate into different functions for easy use and development
    def run_abag_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.stem.replace("_model", "")
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb")
                # trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                summary_confidence_path = os.path.join(refold_path.parent, f"{sample_name}_summary_confidences.json")
                confidence_path = os.path.join(refold_path.parent, f"{sample_name}_confidences.json")
                
                try:
                    ca_rmsd = RMSDCalculator.compute_protein_ca_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                except:
                    ca_rmsd = np.inf
                    print(f"{refold_path} fail for calculate rmsd, set to inf, please check the case")
                plddt, ipae, min_ipae, iptm, ptm_binder, _ptm_H, _ptm_L = Confidence.gather_af3_confidence(
                    confidence_path, summary_confidence_path, inverse_fold_path
                )
                result_data = {
                    'ca_rmsd': ca_rmsd,
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        # Add FoldSeek-based diversity and novelty for antibody-antigen complexes
        df = self._compute_foldseek_diversity_and_novelty(
            pipeline_dir=pipeline_dir,
            df=df,
            output_csv=output_csv,
            subdir_suffix="abag",
        )
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
    
    def run_ligand_binding_protein_evaluation(self, pipeline_dir: str, cands: str, output_csv: str):
        
        def process_metrics_worker(cand: Path):
            try:
                refold_paths = cand.cif_paths
                result_data_all = defaultdict(dict)
                refold_path = refold_paths[0]
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb")
                # trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                plddt, ipae, min_ipae, iptm, ptm_binder = Confidence.gather_chai1_confidence(cand, inverse_fold_path)
                
                result_data_all[f"{sample_name}"] = {
                    'plddt': plddt,
                    'ipae': ipae,
                    'min_ipae': min_ipae,
                    'iptm': iptm,
                    'ptm_binder': ptm_binder,
                }
                return result_data_all
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        cands = pickle.load(open(cands, 'rb'))
        print(f"{len(cands)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for cand in cands:
                future = executor.submit(process_metrics_worker, cand)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(cands), desc="computing metrics"):
                result_data = future.result()
                if result_data is not None:
                    raw_data.update(result_data)

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        # Add FoldSeek-based diversity and novelty for ligand-binding protein designs
        df = self._compute_foldseek_diversity_and_novelty(
            pipeline_dir=pipeline_dir,
            df=df,
            output_csv=output_csv,
            subdir_suffix="ligand_binding_protein",
        )
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
    
    def run_atomic_motif_enzyme_evaluation(
        self,
        pipeline_dir: str,
        cands: str,
        output_csv: str,
        pocket_cutoff: float = 6.0,
        ame_csv: Optional[str] = None,
    ):
        """
        Atomic Motif Enzyme (AME) evaluation.

        Assumes chai-1 refold outputs saved via `ReFold.run_chai1()` to `chai1_cands.pkl`.
        Produces a per-design CSV similar to ligand-binding-protein evaluation, plus:
        - pocket_aligned_ligand_rmsd: ligand RMSD after aligning pocket residues (defined from reference backbone)
        - pocket_residue_count
        - ligand_atoms_matched
        - catalytic_constraints: 6 criteria metrics
        - default: clash detection metrics
        - backbone: ligand distance and secondary structure metrics
        - sidechain: sidechain RMSD metrics
        
        Args:
            pipeline_dir: Root directory of the pipeline
            cands: Path to chai1_cands.pkl file
            output_csv: Output CSV file path
            pocket_cutoff: Cutoff distance for pocket definition (default: 6.0)
            ame_csv: Optional path to AME CSV file (for motif residue information)
        """
        from evaluation.metrics.ame_metrics import (
            compute_catalytic_constraints,
        )
        
        # Load AME CSV if provided
        ame_csv_df = None
        if ame_csv is not None and os.path.exists(ame_csv):
            from inversefold.ame_csv_utils import load_ame_csv, match_pdb_to_csv_info
            ame_csv_df = load_ame_csv(ame_csv)
            print(f"Loaded AME CSV with {len(ame_csv_df)} entries")

        def process_metrics_worker(cand: Path):
            try:
                refold_paths = cand.cif_paths
                refold_path = refold_paths[0]
                sample_name = refold_path.parent.name

                # Convert chai1 directory name to formatted design name
                # Remove the final -{1-8} suffix from seq_0-1-{1-8}
                # Example: m0024_1nzy_seed_42_bb_0_seq_0-1-1 -> m0024_1nzy_seed_42_bb_0_seq_0-1
                import re
                formatted_design_name = re.sub(r'seq_0-1-(\d+)$', 'seq_0-1', sample_name)

                # Reference backbone for pocket definition:
                # Prefer inversefold backbones (contains designed sequence but same coordinates/ligand)
                ref_backbone_path = os.path.join(
                    pipeline_dir, "inverse_fold", "backbones", f"{sample_name}.pdb"
                )
                if not os.path.exists(ref_backbone_path):
                    # Fallback: formatted_designs (raw backbone)
                    # Use formatted_design_name instead of sample_name for formatted_designs
                    # Try formatted_designs in pipeline_dir first, then in parent directory
                    ref_backbone_path = os.path.join(
                        pipeline_dir, "formatted_designs", f"{formatted_design_name}.pdb"
                    )
                    if not os.path.exists(ref_backbone_path):
                        # Try parent directory (for cases where formatted_designs is in ../formatted_designs)
                        parent_dir = os.path.dirname(pipeline_dir)
                        ref_backbone_path = os.path.join(
                            parent_dir, "formatted_designs", f"{formatted_design_name}.pdb"
                        )
                
                # Ensure the path is absolute and exists
                ref_backbone_path = os.path.abspath(ref_backbone_path)
                if not os.path.exists(ref_backbone_path):
                    raise FileNotFoundError(
                        f"Reference backbone file not found. Tried:\n"
                        f"  1. {os.path.join(pipeline_dir, 'inverse_fold', 'backbones', f'{sample_name}.pdb')}\n"
                        f"  2. {os.path.join(pipeline_dir, 'formatted_designs', f'{formatted_design_name}.pdb')}\n"
                        f"  3. {os.path.join(os.path.dirname(pipeline_dir), 'formatted_designs', f'{formatted_design_name}.pdb')}"
                    )
                
                # Design structure (same as ref_backbone_path for AME)
                des_pdb_path = ref_backbone_path

                # Get motif residues from CSV if available
                motif_residues = []
                if ame_csv_df is not None:
                    from inversefold.ame_csv_utils import match_pdb_to_csv_info
                    csv_info = match_pdb_to_csv_info(Path(ref_backbone_path), ame_csv_df)
                    if csv_info is not None:
                        motif_residues = csv_info.get('motif_residues', [])

                # Initialize result dict with only the two required metrics
                result = {
                    "catalytic_heavy_atom_rmsd": np.nan,
                    "ligand_clash_count_1_5A": 0,
                }
                
                # Compute AME-specific metrics if motif residues are available
                if motif_residues and os.path.exists(ref_backbone_path) and os.path.exists(des_pdb_path) and os.path.exists(str(refold_path)):
                    try:
                        # Only compute the two required metrics: catalytic_heavy_atom_rmsd and ligand_clash_count_1_5A
                        # We need to extract these from compute_catalytic_constraints but only keep what we need
                        catalytic_metrics = compute_catalytic_constraints(
                            ref_pdb=ref_backbone_path,
                            des_pdb=des_pdb_path,
                            chai_pdb=str(refold_path),
                            motif_residues=motif_residues,
                            chai_plddt=None,  # Not needed for our metrics
                        )
                        
                        # Only keep the two required metrics
                        result['catalytic_heavy_atom_rmsd'] = catalytic_metrics.get('catalytic_heavy_atom_rmsd', np.nan)
                        result['ligand_clash_count_1_5A'] = catalytic_metrics.get('ligand_clash_count_1_5A', 0)
                    except Exception as e:
                        print(f"ERROR: Failed to compute AME-specific metrics for {sample_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Set default values for required metrics
                        result['catalytic_heavy_atom_rmsd'] = np.nan
                        result['ligand_clash_count_1_5A'] = 0
                else:
                    # No motif residues or missing files - set default values
                    result['catalytic_heavy_atom_rmsd'] = np.nan
                    result['ligand_clash_count_1_5A'] = 0

                return sample_name, result
            except Exception as e:
                import traceback
                try:
                    print(f"ERROR: Error processing {refold_path}: {e}")
                    traceback.print_exc()
                except Exception:
                    print(f"ERROR: Error processing cand: {e}")
                    traceback.print_exc()
                return None, None

        raw_data = defaultdict(dict)
        cands_obj = pickle.load(open(cands, "rb"))
        print(f"{len(cands_obj)} files were found for evaluation.")

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for cand in cands_obj:
                futures.append(executor.submit(process_metrics_worker, cand))
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(cands_obj),
                desc="computing metrics",
            ):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient="index")
        
        # Add success column based on rfd2_success (RFD2 paper standard)
        # success = True if and only if:
        #   - catalytic_heavy_atom_rmsd < 1.5 Å
        #   - ligand_clash_count_1_5A == 0 (no clashes < 1.5 Å)
        if 'rfd2_success' in df.columns:
            df['success'] = df['rfd2_success'].fillna(False).astype(bool)
        else:
            # Fallback: compute success from individual metrics if rfd2_success not available
            catalytic_pass = df['catalytic_heavy_atom_rmsd'].notna() & (df['catalytic_heavy_atom_rmsd'] < 1.5)
            clash_pass = df.get('ligand_clash_count_1_5A', pd.Series(0, index=df.index)) == 0
            df['success'] = catalytic_pass & clash_pass
        
        # Simplify AME output: only keep three columns as requested
        # 1. catalytic_heavy_atom_rmsd
        # 2. ligand_clash_count_1_5A (protein-ligand clash count)
        # 3. success
        columns_to_keep = ['catalytic_heavy_atom_rmsd', 'ligand_clash_count_1_5A', 'success']
        # Ensure all required columns exist, fill missing with NaN/False
        for col in columns_to_keep:
            if col not in df.columns:
                if col == 'success':
                    df[col] = False
                else:
                    df[col] = np.nan
        
        # Keep only the three required columns
        df_simplified = df[columns_to_keep].copy()
        
        # Save simplified CSV
        df_simplified.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
        print(f"Output contains {len(df_simplified)} samples with columns: {list(df_simplified.columns)}")
    
    def run_protein_binding_nuc_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", f"{sample_name}.cif")
                trb_path = os.path.join(pipeline_dir, "formatted_designs", f"{sample_name.rsplit('-',1)[0]}.pkl")
                rmsd = RMSDCalculator.compute_protein_align_nuc_rmsd(pred=str(refold_path), refold=inverse_fold_path, trb=trb_path)
                result_data = {
                    'rmsd': rmsd,
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')
        df.to_csv(output_csv, index=True)
        print(f"metrics computation completed and saved to {output_csv}.")
    
    def run_nuc_evaluation(self, pipeline_dir: str, output_csv: str):
        
        def process_metrics_worker(refold_path: Path):
            try:
                sample_name = refold_path.parent.name
                inverse_fold_path = os.path.join(pipeline_dir, "inverse_fold", f"{sample_name}.cif")
                rmsd = RMSDCalculator.compute_C4_rmsd(pred=str(refold_path), refold=inverse_fold_path)
                tmscore = USalign.compute_tmscore(pred=inverse_fold_path, refold=str(refold_path))
                result_data = {
                    'rmsd': rmsd,
                    'tmscore': tmscore,
                }
                return sample_name, result_data
                
            except Exception as e:
                print(f"Warning: Error processing {refold_path.name}: {e}")
                return None, None
        
        raw_data = defaultdict(dict)
        all_refold_paths = list(Path(os.path.join(pipeline_dir, "refold", "af3_out")).rglob("*_model.cif"))
        print(f"{len(all_refold_paths)} files were found for evaluation.")
        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.metrics.num_workers) as executor:
            for refold_path in all_refold_paths:
                future = executor.submit(process_metrics_worker, refold_path)
                futures.append(future)
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_refold_paths), desc="computing metrics"):
                sample_name, result_data = future.result()
                if sample_name is not None and result_data is not None:
                    raw_data[sample_name] = result_data

        df = pd.DataFrame.from_dict(raw_data, orient='index')

        op_map = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq
        }
        all_condition_series = []
        for metric_name, threshold in self.config.metrics.threshold.items():
            if metric_name not in df.columns:
                print(f"Warning: Metric '{metric_name}' in config does not exist in DataFrame, skipping.")
                continue

        try:
            # 2.1 Parse condition string, e.g., ">/0.45"
            op_str, val_str = threshold.split('/')
            threshold_value = float(val_str)
            
            # 2.2 Get corresponding operator function, e.g., operator.gt
            op_func = op_map[op_str]
            
            # 2.3 Apply operator to entire column (vectorized operation)
            # Example: op_func(df_metrics['metric_1'], 0.45)
            # This returns a boolean Series indicating which rows satisfy the condition
            condition_series = op_func(df[metric_name], threshold_value)
            all_condition_series.append(condition_series)

        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: Cannot parse condition '{threshold}' (metric: {metric_name}). Please check format. Error: {e}")
            # If a condition parsing fails, we can either make all samples fail
            # or skip this condition. Here we create an all-False Series
            all_condition_series.append(pd.Series(False, index=df.index))
        
        if not all_condition_series:
            print("No valid thresholds applied.")
            # If config is empty, can define all as success or all as failure
            success_series = pd.Series(True, index=df.index, name="is_successful")
        else:
            # 3.1 Combine all boolean Series into a new DataFrame
            combined_mask = pd.concat(all_condition_series, axis=1)
            
            # 3.2 Check each row (axis=1) if *all* conditions are True
            success_series = combined_mask.all(axis=1)
        
        df['if_success'] = success_series
        df.to_csv(output_csv, index=True)

        success_backbone_list = defaultdict(list)
        for sample_name in df.index:
            sample_case = sample_name.rsplit('-',1)[0]
            case = sample_name.split('-')[0]
            if sample_case in success_backbone_list:
                continue
            if_success = df.loc[sample_name, 'if_success']
            if if_success:
                success_backbone_list[case].append(f"{sample_case}.cif")
        os.makedirs(os.path.join(pipeline_dir, "cluster"), exist_ok=True)
        os.makedirs(os.path.join(pipeline_dir, "cluster", "success"), exist_ok=True)
        for case, backbones in success_backbone_list.items():
            with open(os.path.join(pipeline_dir, "cluster", "success", f"{case}_success_backbones.list"), 'w') as f:
                f.write('\n'.join(backbones))
        
        USalign.compute_qTMclust_metrics(
            success_dir=os.path.join(pipeline_dir, "cluster", "success"),
            gen_dir=os.path.join(pipeline_dir, "formatted_designs"),
            tm_thresh=self.config.metrics.tm_thres,
        )
    
    def run_motif_scaffolding_evaluation(
        self,
        input_dir: str,
        output_dir: str,
        metadata_file: str = None,
        motif_name: str = None,
        refold_dir: Optional[str] = None,
        skip_pipeline_stages: bool = False,
    ):
        """
        Run motif scaffolding evaluation pipeline.
        
        Assumes standardized input:
        - input_dir: Directory containing PDB files (real residues, not Poly-Ala)
        - metadata_file: Path to scaffold_info.csv with motif_placements
        - motif_name: Name of the motif (for MotifBench reference)
        
        Args:
            input_dir: Directory containing input PDB files
            output_dir: Directory to save evaluation results
            metadata_file: Path to scaffold_info.csv (optional)
            motif_name: Name of the motif (optional, for MotifBench)
            
        Returns:
            dict: Evaluation results dictionary
        """
        from evaluation import get_evaluator
        
        # Get evaluator for motif scaffolding task
        evaluator = get_evaluator("motif_scaffolding", self.config)

        # Unified motif_scaffolding pipeline runs IF/refold as standard stages.
        # In that mode, evaluation should only compute metrics from existing artifacts.
        if skip_pipeline_stages:
            inputs = evaluator.load_inputs(input_dir=input_dir, metadata_file=metadata_file)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if refold_dir is None:
                candidate_dirs = [
                    output_path / "refold" / "esmfold_out",
                    output_path / "refold_output",
                ]
                refold_structures = []
                for candidate in candidate_dirs:
                    if candidate.exists():
                        refold_structures = sorted(candidate.glob("*.pdb"))
                        if refold_structures:
                            break
            else:
                refold_structures = sorted(Path(refold_dir).glob("*.pdb"))

            metrics = evaluator.calculate_metrics(
                input_backbones=inputs["pdbs"],
                refold_structures=refold_structures,
                metadata=inputs["metadata"],
                output_dir=output_path,
                motif_name=motif_name,
                scaffold_info_path=inputs.get("scaffold_info_path"),
            )
            return {
                "metrics": metrics,
                "output_dir": output_path,
            }

        # Backward compatibility path: let the evaluator execute the full pipeline.
        if motif_name and hasattr(evaluator, 'motif_name'):
            evaluator.motif_name = motif_name

        results = evaluator.run_evaluation(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_file=metadata_file,
            motif_name=motif_name,
        )
        return results

    # Antibody developability and antibody-interface evaluation removed (not needed for this benchmark)
