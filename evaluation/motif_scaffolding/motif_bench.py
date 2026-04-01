"""
Motif Scaffolding Evaluation for designbench.

Assumes standardized input:
- PDB files with real residues (not Poly-Ala)
- scaffold_info.csv with motif residue ranges specified
- Format: {sample_num, motif_placements, ...}

Pipeline:
1. Load Inputs
2. Inverse Folding (with motif constraints)
3. Refolding
4. Metrics: scRMSD, motifRMSD, Novelty, Diversity
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd
import os
import shutil
import json

class MotifBenchEvaluator:
    """
    Evaluator for Motif Scaffolding task.
    
    Input Contract:
    - PDB files: Real residues, motif positions already identified
    - scaffold_info.csv: Must contain 'motif_placements' column
      Format: "scaffold_before/chain1/scaffold_middle/chain2/scaffold_after"
      Example: "34/A/70" or "30/A/25/B/30"
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Import analysis modules from motif_scaffolding package
        try:
            from evaluation.motif_scaffolding.analysis import utils as au
            from evaluation.motif_scaffolding.analysis import diversity as du
            from evaluation.motif_scaffolding.analysis import novelty as nu
            self.au = au
            self.du = du
            self.nu = nu
        except ImportError as e:
            self.logger.warning(f"Analysis modules not available: {e}")
            self.au = self.du = self.nu = None
        
        # Get internal paths for motif scaffolding resources
        # Get the directory where this file is located (now in motif_scaffolding/)
        self._module_dir = Path(__file__).parent
        self._motif_scaffolding_dir = self._module_dir
        
        # Scripts directory (internal)
        self._scripts_dir = self._motif_scaffolding_dir / "scripts"
        
        # Motif PDBs directory - can be configured or use default internal resources
        if hasattr(config, 'motif_scaffolding') and hasattr(config.motif_scaffolding, 'motif_pdbs_dir'):
            # Use configured path if provided
            self.motif_pdbs_dir = Path(config.motif_scaffolding.motif_pdbs_dir)
        else:
            # Use default internal resources directory
            self.motif_pdbs_dir = (
                self._motif_scaffolding_dir.parent.parent / "assets" / "motif_scaffolding"
            )
        
        # Create resources directory if it doesn't exist
        self.motif_pdbs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_sample_num_from_name(name: str) -> Optional[int]:
        """Extract trailing sample index from names like 11_3TQB_42 or 11_3TQB_42-7."""
        stem = Path(str(name)).stem
        stem = stem.rsplit("-", 1)[0]
        if "_" not in stem:
            return None
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else None

    @classmethod
    def _pdb_sort_key(cls, path: Path) -> tuple[int, int | str]:
        sample_num = cls._extract_sample_num_from_name(path.stem)
        if sample_num is not None:
            return (0, sample_num)
        return (1, path.name)
    
    def load_inputs(
        self,
        input_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Load inputs with motif-specific validation.
        
        Validates that scaffold_info.csv contains motif_placements.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        pdb_files = sorted(list(input_dir.glob("*.pdb")), key=self._pdb_sort_key)
        if not pdb_files:
            raise ValueError(f"No PDB files found in {input_dir}")

        self.logger.info(f"Found {len(pdb_files)} PDB files in {input_dir}")

        if metadata_file is None:
            metadata_file = input_dir / "scaffold_info.csv"

        metadata_file = Path(metadata_file)
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            if 'sample_num' in metadata.columns:
                metadata = metadata.copy()
                metadata['sample_num'] = pd.to_numeric(metadata['sample_num'], errors='coerce')
                if metadata['sample_num'].notna().all():
                    metadata['sample_num'] = metadata['sample_num'].astype(int)
                    metadata = metadata.sort_values('sample_num').reset_index(drop=True)
            self.logger.info(f"Loaded metadata from {metadata_file}")
            scaffold_info_path = metadata_file
        else:
            self.logger.warning(
                f"Metadata file not found: {metadata_file}. Creating minimal metadata."
            )
            metadata = pd.DataFrame({
                'sample_num': range(len(pdb_files)),
                'pdb_file': [f.name for f in pdb_files]
            })
            scaffold_info_path = None
        
        # Validate motif_placements column
        if 'motif_placements' not in metadata.columns:
            raise ValueError(
                "scaffold_info.csv must contain 'motif_placements' column. "
                "Format: 'scaffold_before/chain/scaffold_after' (e.g., '34/A/70')"
            )
        
        return {
            'pdbs': pdb_files,
            'metadata': metadata,
            'input_dir': input_dir,
            'scaffold_info_path': scaffold_info_path
        }

    def run_inverse_folding(
        self,
        backbone_dir: Union[str, Path],
        output_dir: Union[str, Path],
        motif_constraints: Optional[Dict] = None
    ) -> Path:
        """Run inverse folding to design sequences for backbones."""
        from inversefold.inversefold_api import InverseFold

        backbone_dir = Path(backbone_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        inverse_fold_model = InverseFold(self.config)
        gpu_list = str(self.config.gpus).split(',') if hasattr(self.config, 'gpus') else ['0']

        inverse_fold_model.run(
            action="proteinmpnn_distributed",
            input_dir=backbone_dir,
            output_dir=str(output_dir),
            gpu_list=gpu_list,
            origin_cwd=os.getcwd()
        )

        self.logger.info(f"Inverse folding completed. Output: {output_dir}")
        return output_dir

    def run_refolding(
        self,
        sequences_dir: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> Path:
        """Run refolding to predict structures from sequences."""
        from refold.refold_api import ReFold

        sequences_dir = Path(sequences_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        refold_model = ReFold(self.config)
        refold_input_json = output_dir / "refold_inputs.json"
        refold_model.run(
            action="make_esmfold_json_multi_process",
            backbone_dir=sequences_dir / "backbones",
            output_dir=str(refold_input_json)
        )

        refold_output = output_dir / "refold_output"
        refold_output.mkdir(parents=True, exist_ok=True)

        refold_model.run(
            action="run_esmfold",
            sequences_file_json=str(refold_input_json),
            output_dir=str(refold_output)
        )

        self.logger.info(f"Refolding completed. Output: {refold_output}")
        return refold_output

    def _calculate_base_metrics(
        self,
        input_backbones: List[Path],
        refold_structures: List[Path],
        output_dir: Union[str, Path]
    ) -> pd.DataFrame:
        """Fallback metric calculation when motif analysis modules are unavailable."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, (backbone_path, refold_path) in enumerate(zip(input_backbones, refold_structures)):
            if not refold_path.exists():
                self.logger.warning(f"Refold structure not found: {refold_path}")
                continue

            from evaluation.metrics.rmsd import RMSDCalculator
            sc_rmsd = RMSDCalculator.compute_protein_ca_rmsd(
                pred=str(refold_path),
                refold=str(backbone_path)
            )

            results.append({
                'sample_num': i,
                'sc_rmsd': sc_rmsd,
                'backbone_path': str(backbone_path),
                'refold_path': str(refold_path)
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "metrics.csv", index=False)
        return results_df
    
    def generate_motif_info(
        self,
        scaffold_info_path: Union[str, Path],
        motif_name: str,
        output_dir: Union[str, Path]
    ) -> Path:
        """
        Generate motif_info.csv from scaffold_info.csv.

        This uses internal script to convert scaffold_info to motif_info format.
        """
        import subprocess
        import sys

        scaffold_info_path = Path(scaffold_info_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        motif_pdb_path = self.motif_pdbs_dir / f"{motif_name}.pdb"
        if not motif_pdb_path.exists():
            raise FileNotFoundError(
                f"Motif PDB not found: {motif_pdb_path}\n"
                f"Please ensure motif PDB files are available in: {self.motif_pdbs_dir}\n"
                f"Or configure 'motif_pdbs_dir' in your config file."
            )

        motif_info_path = output_dir / "motif_info.csv"

        script_path = self._scripts_dir / "write_motifInfo_from_scaffoldInfo.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        python_path = None
        if hasattr(self.config, 'motif_scaffolding') and hasattr(self.config.motif_scaffolding, 'python_path'):
            python_path = self.config.motif_scaffolding.python_path
        if python_path and not Path(str(python_path)).exists():
            self.logger.warning(
                f"Configured python_path does not exist: {python_path}. "
                f"Falling back to current interpreter: {sys.executable}"
            )
            python_path = None
        if not python_path:
            python_path = sys.executable

        cmd = [
            python_path,
            str(script_path),
            str(scaffold_info_path),
            str(motif_pdb_path),
            str(motif_info_path)
        ]

        self.logger.info(f"Generating motif_info.csv for {motif_name}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        if not motif_info_path.exists():
            raise RuntimeError(f"Failed to generate motif_info.csv")

        return motif_info_path

    def _extract_motif_segments_from_contig(self, contig: str) -> List[tuple[str, int, int]]:
        """
        Parse motif segments from a contig string.

        Supports motif tokens like:
        - A1-21
        - A7
        - A1-7,A28-79
        while skipping scaffold-length tokens like:
        - 34
        - 20-30
        """
        motif_segments: List[tuple[str, int, int]] = []
        if not contig:
            return motif_segments

        for part in str(contig).split("/"):
            token = part.strip()
            if not token:
                continue
            if not token[0].isalpha():
                # Scaffold token (length/range), not motif residue mapping.
                continue

            for motif_piece in token.split(","):
                piece = motif_piece.strip()
                if not piece:
                    continue
                chain = piece[0]
                residue_part = piece[1:]
                if not residue_part:
                    raise ValueError(f"Invalid motif contig token: '{piece}' in contig '{contig}'")
                if "-" in residue_part:
                    start_text, end_text = residue_part.split("-", 1)
                    start, end = int(start_text), int(end_text)
                else:
                    start = end = int(residue_part)
                motif_segments.append((chain, start, end))
        return motif_segments

    def _parse_contig_with_scaffold_lengths(self, contig: str, split_char: str = "/") -> List[tuple]:
        """
        Parse contig string to extract scaffold lengths and motif segments.
        
        For contig like "52/A1-21/52", returns:
        - (52,) for scaffold length
        - ('motif', 'A', 1, 21) for motif segment
        - (52,) for scaffold length
        
        For contig like "31/A1-15/32/B1-15/32", returns:
        - (31,) for scaffold length
        - ('motif', 'A', 1, 15) for motif segment
        - (32,) for scaffold length
        - ('motif', 'B', 1, 15) for motif segment
        - (32,) for scaffold length
        
        Note: A range like "0-100" as a scaffold token means scaffold length is 100.
        Also supports ";" as split_char for formats like "0-100;A1-21;0-100".
        """
        segments = []
        for part in str(contig).split(split_char):
            token = part.strip()
            if not token:
                continue
            if token[0].isalpha():
                # Motif segment - parse chain and residue range
                for motif_piece in token.split(","):
                    piece = motif_piece.strip()
                    if not piece:
                        continue
                    chain = piece[0]
                    residue_part = piece[1:]
                    if "-" in residue_part:
                        start_text, end_text = residue_part.split("-", 1)
                        start, end = int(start_text), int(end_text)
                    else:
                        start = end = int(residue_part)
                    segments.append(('motif', chain, start, end))
            else:
                # Scaffold length token - could be a single number or range
                if "-" in token:
                    # Range format: "0-100" means scaffold of length 100
                    parts = token.split("-")
                    if len(parts) == 2:
                        try:
                            # If both parts are numbers, it's a range (use end value as length)
                            scaffold_len = int(parts[1])
                        except ValueError:
                            # If not pure numbers, try parsing as before
                            scaffold_len = int(parts[0]) + int(parts[1])
                    else:
                        scaffold_len = int(parts[0])
                else:
                    scaffold_len = int(token)
                segments.append(('scaffold', scaffold_len))
        return segments

    def _build_reference_and_sample_motif_contigs(
        self,
        contig: str,
        segment_order: str
    ) -> tuple[str, str]:
        """
        Build aligned motif-only contigs for reference motif PDB and sampled structure.

        For contig "52/A1-21/52":
        - reference_contig: "A1-21" (motif PDB residue numbering)
        - sample_contig: "A53-73" (scaffold PDB residue numbering, offset by scaffold lengths)
        
        For contig "31/A1-15/32/B1-15/32":
        - reference_contig: "A1-15/B1-15"
        - sample_contig: "A32-46/B78-92"
        """
        parsed_segments = self._parse_contig_with_scaffold_lengths(contig, split_char=';' if ';' in contig else '/')
        if not parsed_segments:
            raise ValueError(f"No segments found in contig: '{contig}'")

        reference_chain_order = [
            chain_id.strip()
            for chain_id in str(segment_order).split(";")
            if chain_id and chain_id.strip()
        ]

        # Walk left-to-right over the linear designed chain.
        # Use a running cursor that accumulates both scaffold and motif lengths.
        motif_idx = 0  # Index into motif segments (for reference chain ordering)
        sample_cursor = 0
        reference_parts = []
        sample_parts = []

        for seg in parsed_segments:
            if seg[0] == 'scaffold':
                # Add scaffold length to running cursor
                sample_cursor += seg[1]
            else:
                # Motif segment
                chain, ref_start, ref_end = seg[1], seg[2], seg[3]
                motif_len = ref_end - ref_start + 1
                
                # Place motif by cumulative designed length, not by reference residue id.
                sample_start = sample_cursor + 1
                sample_end = sample_cursor + motif_len
                
                # Get reference chain from segment_order
                ref_chain = reference_chain_order[motif_idx] if motif_idx < len(reference_chain_order) else chain
                
                # Format reference contig (using motif PDB residue numbering)
                if ref_start == ref_end:
                    reference_parts.append(f"{ref_chain}{ref_start}")
                else:
                    reference_parts.append(f"{ref_chain}{ref_start}-{ref_end}")
                
                # Format sample contig (using scaffold PDB residue numbering)
                if sample_start == sample_end:
                    sample_parts.append(f"{chain}{sample_start}")
                else:
                    sample_parts.append(f"{chain}{sample_start}-{sample_end}")
                
                sample_cursor += motif_len
                motif_idx += 1

        return "/".join(reference_parts), "/".join(sample_parts)

    def _get_structure_chain_ids(self, structure_path: Union[str, Path]) -> List[str]:
        """Return chain IDs present in a PDB structure."""
        chain_ids = []
        with open(structure_path, "r") as handle:
            for line in handle:
                if line.startswith("ATOM"):
                    chain_id = line[21].strip() or " "
                    if chain_id not in chain_ids:
                        chain_ids.append(chain_id)
        return chain_ids

    def _normalize_sample_contig_for_structure(
        self,
        sample_contig: str,
        structure_path: Union[str, Path]
    ) -> str:
        """
        Normalize sample motif contig against the actual chains present in a structure.

        Some packaged benchmark inputs flatten multi-segment motifs onto a single chain
        while `scaffold_info.csv` still labels later motif segments as chain B/C.
        When the sample structure only contains one chain, remap missing motif-chain
        labels onto the sole available chain so motif extraction remains aligned with
        the residue numbering encoded in the contig.
        """
        available_chains = self._get_structure_chain_ids(structure_path)
        if not sample_contig or not available_chains:
            return sample_contig

        if len(available_chains) != 1:
            return sample_contig

        fallback_chain = available_chains[0]
        normalized_parts = []
        changed = False

        for token in str(sample_contig).split("/"):
            token = token.strip()
            if not token:
                continue
            chain_id = token[0]
            if chain_id.isalpha() and chain_id not in available_chains:
                normalized_parts.append(fallback_chain + token[1:])
                changed = True
            else:
                normalized_parts.append(token)

        normalized_contig = "/".join(normalized_parts)
        if changed:
            self.logger.debug(
                "Normalized sample contig to available chain '%s': %s -> %s",
                fallback_chain,
                sample_contig,
                normalized_contig,
            )
        return normalized_contig
    
    def calculate_metrics(
        self,
        input_backbones: List[Path],
        refold_structures: List[Path],
        metadata: pd.DataFrame,
        output_dir: Union[str, Path],
        motif_name: Optional[str] = None,
        scaffold_info_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Calculate motif-specific metrics: scRMSD, motifRMSD, Novelty, Diversity.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.au is None or self.du is None or self.nu is None:
            self.logger.warning("MotifBench analysis modules not available. Using base metrics only.")
            return self._calculate_base_metrics(input_backbones, refold_structures, output_dir)
        
        # Get motif_name from metadata or use default
        if motif_name is None:
            col = metadata.get('motif_name', None)
            motif_name = col.iloc[0] if hasattr(col, 'iloc') and len(metadata) > 0 else 'default_motif'
        
        motif_pdb_path = self.motif_pdbs_dir / f"{motif_name}.pdb"
        if not motif_pdb_path.exists():
            raise FileNotFoundError(f"Motif PDB not found: {motif_pdb_path}")
        
        # Generate motif_info.csv from scaffold_info.csv (MotifBench format)
        scaffold_info_path = scaffold_info_path or metadata.get('scaffold_info_path', None)
        if scaffold_info_path and Path(scaffold_info_path).exists():
            motif_info_path = self.generate_motif_info(
                scaffold_info_path, motif_name, output_dir
            )
            motif_info_df = pd.read_csv(motif_info_path)
        else:
            # Create minimal motif_info from metadata
            motif_info_df = metadata.copy()
        
        results = []
        successful_backbones = []

        # Map design_base (backbone stem) -> (motif_info row, backbone path) for lookup.
        # Do this by sample_num when possible because lexicographic filename ordering
        # (e.g. *_10.pdb before *_2.pdb) otherwise misaligns backbone files with CSV rows.
        design_base_to_row_and_backbone = {}
        sample_num_to_row = {}
        if 'sample_num' in motif_info_df.columns:
            motif_info_df = motif_info_df.copy()
            motif_info_df['sample_num'] = pd.to_numeric(
                motif_info_df['sample_num'], errors='coerce'
            )
            for _, row in motif_info_df.dropna(subset=['sample_num']).iterrows():
                sample_num_to_row[int(row['sample_num'])] = row

        for i, backbone_path in enumerate(input_backbones):
            design_base = Path(backbone_path).stem
            row = None
            sample_num = self._extract_sample_num_from_name(design_base)
            if sample_num is not None:
                row = sample_num_to_row.get(sample_num)
            if row is None and i < len(motif_info_df):
                row = motif_info_df.iloc[i]
            if row is None:
                continue
            design_base_to_row_and_backbone[design_base] = (
                row,
                backbone_path
            )
        
        # Iterate over ALL refold structures (e.g. 5 samples x 8 seqs = 40)
        for refold_path in sorted(refold_structures):
            if not refold_path.exists() or refold_path.suffix.lower() != '.pdb':
                continue
            stem = refold_path.stem
            # Parse "01_1LDB_0-1" -> design_base="01_1LDB_0", seq_idx="1"
            if "-" not in stem:
                continue
            design_base, seq_idx = stem.rsplit("-", 1)
            lookup = design_base_to_row_and_backbone.get(design_base)
            if lookup is None:
                continue
            row, backbone_path = lookup
            sample_num = row['sample_num']
            contig = row.get('contig', '')
            
            # Calculate motif RMSD
            try:
                reference_contig, sample_contig = self._build_reference_and_sample_motif_contigs(
                    contig=contig,
                    segment_order=row.get('segment_order', '')
                )
                sample_contig = self._normalize_sample_contig_for_structure(
                    sample_contig,
                    refold_path,
                )
                
                ref_motif = self.au.motif_extract(
                    reference_contig,
                    str(motif_pdb_path),
                    atom_part="backbone"
                )
                refold_motif = self.au.motif_extract(
                    sample_contig,
                    str(refold_path),
                    atom_part="backbone"
                )

                ref_ca = ref_motif[ref_motif.atom_name == "CA"]
                pred_ca = refold_motif[refold_motif.atom_name == "CA"]
                if len(ref_ca) != len(pred_ca):
                    raise ValueError(
                        "Motif atom count mismatch after contig alignment: "
                        f"reference CA={len(ref_ca)}, sample CA={len(pred_ca)}; "
                        f"reference_contig='{reference_contig}', sample_contig='{sample_contig}', "
                        f"contig='{contig}', segment_order='{row.get('segment_order', '')}'"
                    )
                
                motif_rmsd = self.au.rmsd(ref_motif, refold_motif)
                
                # Calculate scRMSD (self-consistency)
                from evaluation.metrics.rmsd import RMSDCalculator
                if backbone_path and Path(backbone_path).exists():
                    sc_rmsd = RMSDCalculator.compute_protein_ca_rmsd(
                        pred=str(refold_path),
                        refold=str(backbone_path)
                    )
                else:
                    sc_rmsd = None
                
                # Success threshold: motif RMSD < 2.0 Å
                success = motif_rmsd < 2.0
                
                results.append({
                    'sample_num': sample_num,
                    'seq_idx': seq_idx,
                    'motif_rmsd': motif_rmsd,
                    'sc_rmsd': sc_rmsd,
                    'success': success,
                    'backbone_path': str(backbone_path) if backbone_path is not None else None,
                    'refold_path': str(refold_path)
                })
                
                if success:
                    successful_backbones.append(str(backbone_path))
                    
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.logger.error(f"Error calculating metrics for {refold_path.name}: {e}\n{tb}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Always write summary artifacts, even when there are zero successful samples.
        self._calculate_diversity_and_novelty(
            successful_backbones, motif_name, output_dir, results_df
        )
        
        results_df.to_csv(output_dir / "motif_metrics.csv", index=False)
        return results_df

    def run_evaluation(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        motif_name: Optional[str] = None
    ) -> Dict:
        """Run complete motif scaffolding evaluation pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Step 1: Loading inputs...")
        inputs = self.load_inputs(input_dir, metadata_file)

        self.logger.info("Step 2: Running inverse folding...")
        inverse_fold_output = output_dir / "inverse_fold"
        self.run_inverse_folding(
            backbone_dir=input_dir,
            output_dir=inverse_fold_output
        )

        self.logger.info("Step 3: Running refolding...")
        refold_output = output_dir / "refold"
        self.run_refolding(
            sequences_dir=inverse_fold_output,
            output_dir=refold_output
        )

        self.logger.info("Step 4: Calculating metrics...")
        metrics = self.calculate_metrics(
            input_backbones=inputs['pdbs'],
            refold_structures=list((refold_output / "refold_output").glob("*.pdb")),
            metadata=inputs['metadata'],
            output_dir=output_dir / "metrics",
            motif_name=motif_name,
            scaffold_info_path=inputs.get('scaffold_info_path')
        )

        return {
            'metrics': metrics,
            'output_dir': output_dir
        }
    
    def _calculate_diversity_and_novelty(
        self,
        successful_backbones: List[str],
        motif_name: str,
        output_dir: Path,
        results_df: pd.DataFrame
    ):
        """Calculate diversity and novelty metrics."""
        output_dir = Path(output_dir)
        successful_dir = output_dir / "successful_backbones"
        successful_dir.mkdir(parents=True, exist_ok=True)

        unique_successful_backbones = sorted(set(successful_backbones))
        
        for backbone_path in unique_successful_backbones:
            shutil.copy2(backbone_path, successful_dir / Path(backbone_path).name)
            # Replace UNK with GLY for foldseek compatibility
            pdb_file = successful_dir / Path(backbone_path).name
            with open(pdb_file, 'r') as f:
                content = f.read()
            content = content.replace('UNK', 'GLY')
            with open(pdb_file, 'w') as f:
                f.write(content)
        
        diversity_result = {'Diversity': 0, 'Clusters': 0, 'Alpha5_Clusters': 0}
        if unique_successful_backbones:
            # Calculate diversity with alpha=5
            foldseek_db = self.config.motif_scaffolding.foldseek_database
            foldseek_bin = os.environ.get("FOLDSEEK_BIN")
            foldseek_db_path = Path(str(foldseek_db))
            if foldseek_db_path.is_dir():
                candidate_prefix = foldseek_db_path / "pdb"
                if candidate_prefix.exists() or candidate_prefix.with_suffix(".dbtype").exists():
                    self.logger.info(
                        f"Resolved foldseek database directory to prefix: {candidate_prefix}"
                    )
                    foldseek_db = str(candidate_prefix)
                else:
                    self.logger.warning(
                        f"Foldseek database path is a directory without 'pdb' prefix: {foldseek_db_path}"
                    )
            assist_protein = self.motif_pdbs_dir / f"{motif_name}.pdb"
            diversity_result = self._calculate_diversity_with_alpha5(
                successful_dir, assist_protein, foldseek_db, foldseek_bin
            )
        
        # Calculate novelty
        novelty_value = 0.0
        success_results = results_df[results_df['success'] == True].copy() if 'success' in results_df.columns else pd.DataFrame()
        if len(success_results) > 0:
            # Prepare dataframe for novelty calculation
            # Novelty calculation expects 'backbone_path' column
            novelty_input = success_results.copy()
            if 'backbone_path' not in novelty_input.columns:
                if 'refold_path' in novelty_input.columns:
                    novelty_input['backbone_path'] = novelty_input['refold_path']
                else:
                    self.logger.warning("Cannot calculate novelty: missing refold_path column")
                    novelty_input = None
            
            if novelty_input is not None and len(novelty_input) > 0:
                try:
                    novelty_results = self.nu.calculate_novelty(
                        input_csv=novelty_input,
                        foldseek_database_path=foldseek_db,
                        max_workers=4,
                        cpu_threshold=75.0,
                        foldseek_path=foldseek_bin,
                    )
                    novelty_path = output_dir / "novelty_results.csv"
                    novelty_results.to_csv(novelty_path, index=False)
                    # Calculate mean novelty (pdbTM column)
                    if 'pdbTM' in novelty_results.columns:
                        # Filter out NaN values
                        pdbTM_values = novelty_results['pdbTM'].dropna()
                        if len(pdbTM_values) > 0:
                            novelty_value = float(pdbTM_values.mean())
                    elif 'novelty' in novelty_results.columns:
                        novelty_values = novelty_results['novelty'].dropna()
                        if len(novelty_values) > 0:
                            novelty_value = float(novelty_values.mean())
                except Exception as e:
                    self.logger.warning(f"Failed to calculate novelty: {e}")
        
        # Count unique solutions (unique successful backbones)
        num_unique_solutions = int(diversity_result.get('Clusters', len(unique_successful_backbones)))
        
        # Calculate success rate
        if 'backbone_path' in results_df.columns and len(results_df) > 0:
            total_scaffolds_evaluated = int(results_df['backbone_path'].nunique())
        else:
            total_scaffolds_evaluated = len(results_df)
        successful_scaffolds = len(unique_successful_backbones)
        success_rate = successful_scaffolds / total_scaffolds_evaluated if total_scaffolds_evaluated > 0 else 0
        
        # Save summary (JSON format)
        summary = {
            'motif_name': motif_name,
            'total_samples': total_scaffolds_evaluated,
            'successful_samples': successful_scaffolds,
            'success_rate': success_rate,
            'diversity': diversity_result.get('Diversity', 0),
            'num_clusters': diversity_result.get('Clusters', 0),
            'num_solutions': num_unique_solutions,
            'alpha5_clusters': diversity_result.get('Alpha5_Clusters', 0),
            'novelty': novelty_value
        }
        
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate esm_summary.txt (MotifBench format)
        esm_summary_path = output_dir / "esm_summary.txt"
        with open(esm_summary_path, 'w') as f:
            f.write(f"Evaluated Protein | {motif_name}\n")
            f.write(f"Number of Unique Solutions (unique successful backbones) | {num_unique_solutions}\n")
            f.write(f"Novelty | {novelty_value:.4f}\n")
            f.write(f"Success Rate | {success_rate * 100:.2f}\n")
            f.write(f"Number of Scaffolds Evaluated | {total_scaffolds_evaluated}\n")
        
        self.logger.info(f"Evaluation complete: {summary}")
    
    def _calculate_diversity_with_alpha5(
        self,
        successful_dir: Path,
        assist_protein: Path,
        foldseek_db: str,
        foldseek_bin: Optional[str] = None,
    ) -> Dict:
        """Calculate diversity using alpha=5 saturation curve."""
        if self.du is None:
            return {'Diversity': 0, 'Clusters': 0, 'Samples': 0, 'Alpha5_Clusters': 0}
        
        target_clusters = 5
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_result = None
        best_diff = float('inf')
        
        for threshold in thresholds:
            try:
                result = self.du.foldseek_cluster(
                    input=str(successful_dir),
                    assist_protein_path=str(assist_protein),
                    tmscore_threshold=threshold,
                    alignment_type=1,
                    output_mode="DICT",
                    save_tmp=True,
                    foldseek_path=foldseek_bin,
                )
                
                num_clusters = result.get('Clusters', 0)
                diff = abs(num_clusters - target_clusters)
                
                if diff < best_diff:
                    best_diff = diff
                    best_result = result
                    best_result['Alpha5_Clusters'] = num_clusters
                    best_result['Alpha5_Threshold'] = threshold
                
                if diff <= 1:
                    break
            except Exception as e:
                self.logger.warning(f"Error clustering with threshold {threshold}: {e}")
                continue
        
        if best_result is None:
            best_result = self.du.foldseek_cluster(
                input=str(successful_dir),
                assist_protein_path=str(assist_protein),
                tmscore_threshold=0.6,
                alignment_type=1,
                output_mode="DICT",
                save_tmp=True,
                foldseek_path=foldseek_bin,
            )
            best_result['Alpha5_Clusters'] = best_result.get('Clusters', 0)
            best_result['Alpha5_Threshold'] = 0.6
        
        return best_result
