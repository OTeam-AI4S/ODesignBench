"""
Orchestrator for motif scaffolding evaluation in designbench.

Handles design_dir layout (per-motif subdirs or single motif dir),
optional Generator (RFD3/PPIFlow), and delegates to MotifBenchEvaluator.
"""
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import pandas as pd
import subprocess
import sys
import logging

from evaluation import get_evaluator


def _discover_motifs(design_dir: Path) -> List[str]:
    """Discover motif names: subdirs that contain *.pdb and scaffold_info.csv."""
    motifs = []
    for sub in sorted(design_dir.iterdir()):
        if not sub.is_dir():
            continue
        pdbs = list(sub.glob("*.pdb"))
        scaffold_info = sub / "scaffold_info.csv"
        if pdbs and scaffold_info.exists():
            motifs.append(sub.name)
    return motifs


def _prepare_input_dir(
    source_dir: Path,
    dest_dir: Path,
    max_samples: Optional[int],
    motif_name: str,
) -> Path:
    """
    Copy up to max_samples PDBs and scaffold_info rows to dest_dir.
    If max_samples is None, symlink or use source_dir (caller can pass source_dir as dest).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdbs = sorted(source_dir.glob("*.pdb"))
    scaffold_info = source_dir / "scaffold_info.csv"

    if max_samples is not None:
        pdbs = pdbs[: max_samples]
        dest_dir = Path(dest_dir)
        for p in pdbs:
            shutil.copy2(p, dest_dir / p.name)
        if scaffold_info.exists():
            df = pd.read_csv(scaffold_info)
            df = df.head(max_samples)
            df.to_csv(dest_dir / "scaffold_info.csv", index=False)
        else:
            # Minimal metadata
            pd.DataFrame({"sample_num": range(len(pdbs))}).to_csv(
                dest_dir / "scaffold_info.csv", index=False
            )
        return dest_dir

    # No limit: copy or symlink all
    for p in pdbs:
        shutil.copy2(p, dest_dir / p.name)
    if scaffold_info.exists():
        shutil.copy2(scaffold_info, dest_dir / "scaffold_info.csv")
    else:
        pd.DataFrame({"sample_num": range(len(pdbs))}).to_csv(
            dest_dir / "scaffold_info.csv", index=False
        )
    return dest_dir


class MotifScaffoldingEvaluation:
    """
    Runs the full motif scaffolding pipeline for one or more motifs.

    design_dir can be:
    - Parent of per-motif subdirs (e.g. rfd3_outputs/01_1LDB, 02_1ITU) -> motif_list or auto-discover
    - Single motif dir (e.g. 01_1LDB with *.pdb and scaffold_info.csv) -> one motif
    """

    def __init__(self, config, model_name: Optional[str] = None):
        self.config = config
        self.model_name = model_name or getattr(
            config, "model_name", None
        ) or config.get("model_name")
        self.logger = logging.getLogger(__name__)

    def run_motif_scaffolding_evaluation(
        self,
        design_dir: str,
        pipeline_dir: str,
        motif_list: Optional[List[str]] = None,
        max_samples_per_motif: Optional[int] = None,
    ) -> Dict[str, Path]:
        """
        Run evaluation for each motif.

        Args:
            design_dir: Path to design outputs (parent of motif subdirs or single motif dir).
            pipeline_dir: Root output directory (e.g. results/).
            motif_list: Optional list of motif names (e.g. [01_1LDB]). If None, auto-discover or use single dir.
            max_samples_per_motif: If set, only run on first N samples per motif (for quick tests).

        Returns:
            Dict mapping motif_name -> result output_dir path.
        """
        design_dir = Path(design_dir)
        pipeline_dir = Path(pipeline_dir)
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Single motif dir: one folder with PDBs + scaffold_info.csv
        pdbs_in_root = list(design_dir.glob("*.pdb"))
        scaffold_in_root = (design_dir / "scaffold_info.csv").exists()

        if pdbs_in_root and scaffold_in_root:
            motifs_to_run = [motif_list[0]] if motif_list and len(motif_list) > 0 else [design_dir.name]
            input_dirs = {m: design_dir for m in motifs_to_run}
        else:
            # Parent of motif subdirs
            if motif_list:
                motifs_to_run = [str(m) for m in motif_list]
            else:
                motifs_to_run = _discover_motifs(design_dir)
            if not motifs_to_run:
                raise ValueError(
                    f"No motif subdirs (with *.pdb and scaffold_info.csv) found in {design_dir}. "
                    "Pass motif_list or use a single motif dir with PDBs and scaffold_info.csv."
                )
            input_dirs = {m: design_dir / m for m in motifs_to_run}
            for m, d in list(input_dirs.items()):
                if not d.exists():
                    raise FileNotFoundError(f"Motif dir not found: {d}")

        evaluator = get_evaluator("motif_scaffolding", self.config)
        results = {}

        for motif_name in motifs_to_run:
            source = input_dirs[motif_name]
            out_dir = pipeline_dir / "motif_scaffolding" / motif_name

            if max_samples_per_motif is not None:
                work_dir = pipeline_dir / "motif_scaffolding" / f"{motif_name}_input"
                work_dir.mkdir(parents=True, exist_ok=True)
                input_dir = _prepare_input_dir(
                    source, work_dir, max_samples_per_motif, motif_name
                )
            else:
                input_dir = source

            metadata_file = input_dir / "scaffold_info.csv"
            evaluator.run_evaluation(
                input_dir=input_dir,
                output_dir=out_dir,
                metadata_file=metadata_file if metadata_file.exists() else None,
                motif_name=motif_name,
            )
            results[motif_name] = out_dir

        # Generate summary files after all evaluations complete
        self._generate_summaries(pipeline_dir)

        return results
    
    def _generate_summaries(self, pipeline_dir: Path):
        """
        Generate summary files: summary_by_problem.csv and overall_summary.csv.
        
        Args:
            pipeline_dir: Root output directory containing motif evaluation results
        """
        try:
            script_path = Path(__file__).parent / "scripts" / "write_summaries.py"
            if not script_path.exists():
                self.logger.warning(f"Summary script not found: {script_path}")
                return
            
            python_path = None
            if hasattr(self.config, 'motif_scaffolding') and hasattr(self.config.motif_scaffolding, 'python_path'):
                python_path = self.config.motif_scaffolding.python_path
            if not python_path:
                python_path = sys.executable
            
            # Try to find test_cases.csv (optional)
            test_cases_path = None
            # Check common locations
            possible_test_cases_paths = [
                Path(__file__).parent.parent.parent / "Motif_Benchmark" / "MotifBench" / "test_cases.csv",
                Path(__file__).parent / "resources" / "test_cases.csv",
                # Common local sibling checkout layout
                Path(__file__).resolve().parents[4] / "MotifBench" / "test_cases.csv",
                # If ODesignBench is under .../hzh/ODesignBench, probe .../hzh/MotifBench
                Path(__file__).resolve().parents[5] / "MotifBench" / "test_cases.csv",
            ]
            for path in possible_test_cases_paths:
                if path.exists():
                    test_cases_path = path
                    break
            
            cmd = [python_path, str(script_path), str(pipeline_dir)]
            if test_cases_path:
                cmd.extend(["--test-cases", str(test_cases_path)])
            
            self.logger.info("Generating summary files...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.logger.info("Summary files generated successfully")
                if result.stdout:
                    self.logger.info(result.stdout)
            else:
                self.logger.warning(f"Summary generation had warnings: {result.stderr}")
                
        except Exception as e:
            self.logger.warning(f"Failed to generate summary files: {e}")
            # Don't raise - summaries are optional
