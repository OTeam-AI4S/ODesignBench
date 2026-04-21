# ODesignBench

ODesignBench is a multimodal benchmark toolkit for structure-based biomolecular design. It standardizes data contracts, inverse-folding/refolding workflows, and evaluation metrics across multiple design settings.

> Note: Additional modules are still being updated and will be released in upcoming updates.

The repository is designed to evaluate generated structures from external models under a consistent protocol, so different generators can be compared fairly with shared preprocessing, folding, and metric pipelines.

## Environment Setup

This project integrates multiple heavy-weight structural biology models (Chai-1, ESM, Protenix, and AlphaFold3). Due to complex C++ dependencies, environment setup is supported via Conda.

A unified `environment.yml` is provided in the repository root. It combines the RosettaCommons channel, NVIDIA CUDA packages, and the PyTorch ecosystem to match the CUDA and runtime assumptions used by the benchmark pipelines.

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/OTeam-AI4S/ODesignBench.git
cd ODesignBench
```

1. Create the Conda environment:

```bash
conda env create -f environment.yml
```

1. Activate the environment:

```bash
conda activate designbench
```

### Notes

- `fair-esm==2.0.0` is used and is compatible with modern Python 3.10 and PyTorch 2.x.
- `openbabel` is included in `environment.yml` for ligand reconstruction/docking metrics (`mol_rec`, `rdkit_utils`, `docking_vina`).
- `environment.yml` also includes the extra ligand-evaluation stack used by `run_pbl_pipeline.py`, including `EFGs`, `meeko==0.1.dev3`, `pdb2pqr`, `vina`, and `AutoDockTools_py3`.
- If your system has multiple CUDA installations, verify runtime visibility with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.

### Verifying Ligand Docking Dependencies

After environment creation, it is helpful to verify the docking stack before launching a large PBL run:

```bash
conda activate designbench
python -c "import AutoDockTools, vina, meeko, easydict, EFGs; print('PBL docking dependencies are available')"
pdb2pqr30 --help
```

If your environment was created before these dependencies were added to `environment.yml`, recreate the environment or install the missing packages into the existing `designbench` environment.

## Inversefold Model Parameters (ProteinMPNN / LigandMPNN)

Pipelines that use `inversefold=ProteinMPNN` require the ProteinMPNN checkpoint at:

- `inversefold/LigandMPNN/model_params/proteinmpnn_v_48_020.pt`

If this file is missing, you may see:

- `FileNotFoundError: ProteinMPNN checkpoint not found: .../proteinmpnn_v_48_020.pt`

Download model params into the expected repo-local directory:

```bash
bash inversefold/LigandMPNN/get_model_params.sh inversefold/LigandMPNN/model_params
```

By default, the config reads:

- `PROTEINMPNN_CHECKPOINT_PATH` (environment variable), or
- `inversefold.checkpoint_path` (Hydra override, defaulting to the path above)

Optional explicit override examples:

```bash
export PROTEINMPNN_CHECKPOINT_PATH=/absolute/path/to/proteinmpnn_v_48_020.pt
python scripts/run_pbp_pipeline.py design_dir=/path/to/pbp_designs gpus=0
```

```bash
python scripts/run_pbp_pipeline.py \
  design_dir=/path/to/pbp_designs \
  gpus=0 \
  inversefold.checkpoint_path=/absolute/path/to/proteinmpnn_v_48_020.pt
```

## ESMFold Weights

To run ESMFold for refolding, you need to download the ESMFold model weights from Hugging Face. The scripts expect the weights to be located in `refold/esmfold/weights`.

Download using the Python API (recommended):

```bash
# Make sure you are in the ODesignBench directory
mkdir -p refold/esmfold/weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esmfold_v1', local_dir='refold/esmfold/weights')"
```

Or manually download the weights from [Hugging Face](https://huggingface.co/facebook/esmfold_v1/tree/main).

## Chai-1 Weights

By default, the `chai_lab` library attempts to download its model weights and ESM weights automatically during the first run. To bypass this, you may download the weights before running ODesignBench pipeline: 

### Method 1: Fast Download via `aria2` (Recommended for unstable networks)

You can use a multi-threaded download tool like `aria2` to download all weights directly from their source URLs. This is much faster and more reliable than the default Python script.

```bash
# 1. Define where you want to store the weights
export CHAI_DOWNLOADS_DIR=$(pwd)/refold/chai1/weights
mkdir -p $CHAI_DOWNLOADS_DIR/models_v2
mkdir -p $CHAI_DOWNLOADS_DIR/esm

# 2. Download Chai-1 main components
for comp in feature_embedding.pt token_embedder.pt trunk.pt diffusion_module.pt confidence_head.pt; do
    aria2c -x 16 -s 16 -d $CHAI_DOWNLOADS_DIR/models_v2 -o $comp "https://chaiassets.com/chai1-inference-depencencies/models_v2/$comp"
done

# 3. Download Conformers
aria2c -x 16 -s 16 -d $CHAI_DOWNLOADS_DIR -o conformers_v1.apkl "https://chaiassets.com/chai1-inference-depencencies/conformers_v1.apkl"

# 4. Download ESM weights
aria2c -x 16 -s 16 -d $CHAI_DOWNLOADS_DIR/esm -o traced_sdpa_esm2_t36_3B_UR50D_fp16.pt "https://chaiassets.com/chai1-inference-depencencies/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
```

### Method 2: Pre-download via Python script

Run the following Python script:

```python
import os
from chai_lab.utils.paths import chai1_component, cached_conformers

# Define where you want to store the weights
download_dir = os.path.abspath("./refold/chai1/weights")
os.environ["CHAI_DOWNLOADS_DIR"] = download_dir
os.makedirs(download_dir, exist_ok=True)

components = [
    "feature_embedding.pt", 
    "token_embedder.pt", 
    "trunk.pt", 
    "diffusion_module.pt", 
    "confidence_head.pt"
]

print(f"Downloading Chai-1 weights to {download_dir}...")
for comp in components:
    print(f"Fetching {comp}...")
    chai1_component(comp)

print("Fetching conformers_v1.apkl...")
cached_conformers.get_path()

print("Fetching ESM weights...")
from chai_lab.data.dataset.embeddings.esm import ESM_URL, esm_cache_folder
from chai_lab.utils.paths import download_if_not_exists
local_esm_path = esm_cache_folder.joinpath("traced_sdpa_esm2_t36_3B_UR50D_fp16.pt")
download_if_not_exists(ESM_URL, local_esm_path)
    
print("✅ Chai-1 and ESM weights successfully downloaded!")
```

### Next Step: Set the Environment Variable

Regardless of whether you used Method 1 or Method 2, before running the ODesignBench pipeline on your compute node, export the `CHAI_DOWNLOADS_DIR` variable to tell the pipeline where the offline weights are located:

```Bash
# Replace with your actual absolute path
export CHAI_DOWNLOADS_DIR="/path/to/ODesignBench/refold/chai1/weights"

# Now you can safely run the pipeline without auto-download
python3 scripts/run_ame_pipeline.py design_dir=examples/tip_atom_scaffolding/ gpus=0 
```

## AlphaFold3 Setup

Some benchmark tasks use `refold=af3` and run AlphaFold3 through a wrapper script.
Default wrapper: `refold/run_af3.sh` (Docker).
HPC wrapper: `refold/run_af3_singularity.sh` (Singularity/Apptainer).

### Download required artifacts

Follow the official AlphaFold3 instructions in the upstream repo to download:

1. AF3 model parameters (see: [Obtaining model parameters](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters))
2. AF3 public databases (see the database download instructions in the upstream AlphaFold3 repo: [https://github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)).

### Directory layout expected by `refold/run_af3.sh`

The wrapper expects:

- `$AF3_BASE/models` -> mounted to `/root/models` inside the container
- `$AF3_PUBLIC_DB` -> mounted to `/root/public_databases` inside the container

### Configure environment variables

Set these before running any pipeline that uses AF3 refolding:

```bash
export AF3_BASE=/path/to/af3              # must contain: $AF3_BASE/models
export AF3_PUBLIC_DB=/path/to/public_databases
# Optional: if your docker image tag differs
export AF3_DOCKER_IMAGE=alphafold3
```

### Singularity/Apptainer mode

Singularity/Apptainer is commonly allowed in HPC and can run containerized AF3 workloads without Docker daemon privileges.
If your cluster does not support Docker on compute nodes, switch AF3 execution to:

```bash
export AF3_EXEC=/absolute/path/to/ODesignBench/refold/run_af3_singularity.sh
export AF3_SIF_IMAGE=/absolute/path/to/alphafold3.sif
export AF3_BASE=/path/to/af3              # must contain: $AF3_BASE/models
export AF3_PUBLIC_DB=/path/to/public_databases
```

`run_af3_singularity.sh` accepts both `singularity` and `apptainer` commands.
If your command name differs, ensure it is available in `PATH`.

**Note on PBP target MSA injection:** PBP tasks inject pre-computed MSA for the target chain via a runtime patch (`AF3_DIALECT_PATCH=true`, default). The assets are expected at `/assets`. Set `AF3_DIALECT_PATCH=false` to disable the patch.

### Run

After exporting the variables above, run the normal pipeline commands (e.g. `scripts/run_pbp_pipeline.py` with `refold=af3`).

## Benchmark Content

### Nucleic Acid Monomer (DNA / RNA)

`run_nuc_pipeline.py` evaluates nucleic-acid designs (DNA or RNA) generated by external models (for example ODesign).

- Input: nucleic-acid structure files (`.cif`/`.pdb`) in `design_dir`
- Expected input contract: `nuc` currently assumes the input directory already contains inverse-fold outputs (i.e. sequence-designed structures ready for refolding/evaluation)
- Inverse fold: not re-run for `nuc` task inside ODesignBench; input structures are copied into `inverse_fold/` and treated as the post-inversefold reference structures
- Refold: AlphaFold3
- Evaluation: C4' RMSD and TM-score

Minimal run command:

```bash
python scripts/run_nuc_pipeline.py \
  design_dir=/path/to/nuc_designs \
  refold.run_data_pipeline=true \
  gpus=0
```

Optional: set `root=/path/to/output_dir` to change output location (default: `results`).

Example:

```bash
python3 scripts/run_nuc_pipeline.py \
  design_dir=examples/nuc/rna \
  refold.run_data_pipeline=true \
  gpus=0 \
  root=results/examples/nuc
```

For RNA without precomputed MSA in AF3 input JSON, set `refold.run_data_pipeline=true` to let AF3 run MSA/template search during refold. Keep global default `run_data_pipeline=false` for other tasks.

### Atomic Motif Scaffolding / Enzyme Design (AME)

AME evaluates the scaffolding of atomic motifs, which are crucial for enzyme design and small molecule binding:

- Input: scaffold PDB files + `ame_info.csv` (or `ame.csv`)
- Inverse fold: LigandMPNN (motif-constrained design)
- Refold: Chai-1
- Evaluation: catalytic heavy atom RMSD, ligand clash count, overall success rate

Minimal run command:

```bash
python scripts/run_ame_pipeline.py \
  design_dir=/path/to/ame_scaffolds \
  gpus=0
```

Optional: set `root=/path/to/output_dir` to change output location (default: `results`).

`ame.csv` (or `ame_input.csv`) should include 3 columns (no header required):

1. `id`: The filename of the scaffold (e.g., `m0024_1nzy_seed_46_bb_9_seq_0-1.pdb`)
2. `task`: The AME task name (one of the 41 standard tasks, e.g., `M0024_1nzy`)
3. `motif_residues`: Comma-separated list of motif residues to keep fixed (e.g., `"A114,A137,A145,A64,A86,A90"`)

Example:

```csv
m0024_1nzy_seed_46_bb_9_seq_0-1.pdb,M0024_1nzy,"A114,A137,A145,A64,A86,A90"
```

### Protein Binding Protein (PBP)

PBP evaluates designed protein-protein complexes with a fixed interface role definition:

- Input: complex PDB files in `design_dir`
- Required metadata: `pbp_info.csv` in `design_dir`
- Inverse fold: LigandMPNN
- Refold: AlphaFold3 (sequence-only)
- Evaluation: interface and structure quality metrics in the benchmark pipeline

Minimal run command:

```bash
python scripts/run_pbp_pipeline.py \
  design_dir=/path/to/pbp_designs \
  gpus=0
```

Optional: set `root=/path/to/output_dir` to change output location (default: `results`).

`pbp_info.csv` must provide one row per design with at least:

- `design_name`
- `design_chain`
- `target_chain`

Example:

```csv
design_name,target_chain,design_chain
CD3d_0,B,A
CD3d_1,B,A
```

### Ligand Binding Protein (LBP)

LBP evaluates ligand-binding protein designs, where the ligand is retained during inverse folding and the designed protein is refolded with Chai-1.

- Input: ligand-containing design structures (`.cif` recommended)
- Required metadata: `lbp_info.csv` in `design_dir`
- Recommended layout: nested case folders such as `design_dir/FAD/FAD_seed_1_bb_0_seq_0.cif`
- Inverse fold: LigandMPNN
- Refold: Chai-1
- Evaluation: `plddt`, `ipae`, `min_ipae`, `iptm`, `ptm_binder`, plus Foldseek-based diversity/novelty

LBP uses the CCD components database during ligand handling. By default the repo expects:

- `preprocess/ccd_component/components.cif`

If this file is missing, download it from wwPDB and place it there:

```bash
cd /path/to/ODesignBench/preprocess/ccd_component
wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
gunzip -f components.cif.gz
```

Minimal run command:

```bash
python scripts/run_lbp_pipeline.py \
  design_dir=/path/to/ligand_binding_protein_designs \
  gpus=0
```

Optional: set `root=/path/to/output_dir` to change output location (default: `results`).

`lbp_info.csv` must provide one row per design with at least:

- `design_name`
- `target_chain`
- `design_chain`

Example:

```csv
design_name,target_chain,design_chain
FAD_seed_1_bb_0_seq_0,A,B
```

Recommended example layout:

```text
examples/ligand_binding_protein/
|-- lbp_info.csv
`-- FAD/
    `-- FAD_seed_1_bb_0_seq_0.cif
```

Example run:

```bash
python3 scripts/run_lbp_pipeline.py \
  design_dir=examples/ligand_binding_protein \
  gpus=0 \
  root=results/examples/ligand_binding_protein
```

Successful runs will write the final evaluation table to `raw_data.csv`.

### Interface Design

Interface design is the pocket-constrained variant of ligand-binding protein design used in the ODesign paper. It reuses the same protein-ligand inputs as LBP, but only redesigns protein residues within `3.5A` of the ligand.

- Input: ligand-containing design structures (`.cif` recommended)
- Required metadata: `lbp_info.csv` in `design_dir`
- Recommended layout: nested case folders such as `design_dir/FAD/FAD_seed_1_bb_0_seq_0.cif`
- Pocket definition: protein residues within `3.5A` of any non-water ligand atom
- Inverse fold: LigandMPNN with pocket-only redesign
- Refold: Chai-1
- Evaluation: `sc_rmsd`, `pocket_rmsd`, pocket `plddt`, plus `global_plddt`, `ipae`, `min_ipae`, `iptm`, `ptm_binder`

The unified interface pipeline now has its own config:

```bash
python3 scripts/run_interface_pipeline.py \
  design_dir=examples/ligand_binding_protein \
  gpus=0 \
  root=results/examples/interface_design
```

`interface` reads the same `lbp_info.csv` as `lbp` to determine the ligand `target_chain` and redesigned `design_chain`, then computes the pocket only on `design_chain`.

### Protein Binding Nucleic Acid (PBN)

PBN evaluates designed protein-nucleic acid complexes, where the protein chain is treated as the fixed conditioning partner and the nucleic-acid chain is the designed partner.

- Input: sequence-assigned complex structures (`.cif` recommended)
- Required metadata: `pbn_info.csv`
- Current validated input contract: the input structures are already post-inversefold outputs (for example ODesign `*_bb_*_seq_*.cif`)
- Inverse fold: skipped inside ODesignBench for this workflow; inputs are copied into `inverse_fold/` and treated as the post-inversefold reference structures
- Refold: AlphaFold3
- Evaluation: protein-aligned nucleic-acid `C4'` RMSD

`pbn_info.csv` must provide one row per design with at least:

- `design_name`
- `target_chain`
- `design_chain`

Example:

```csv
design_name,target_chain,design_chain
prot_binding_rna_demo_seed_2_bb_0_seq_0,A,B
```

Recommended example layout:

```text
examples/protein_binding_nuc/
|-- pbn_info.csv
`-- prot_binding_rna_demo_seed_2_bb_0_seq_0.cif
```

Example run:

```bash
python3 scripts/run_pbn_pipeline.py \
  design_dir=examples/protein_binding_nuc \
  gpus=0 \
  root=results/examples/protein_binding_nuc
```

Successful runs will write:

- preprocessed inputs to `formatted_designs/`
- copied post-inversefold references to `inverse_fold/`
- AlphaFold3 outputs to `refold/af3_out/`
- final metrics to `raw_data.csv`

The current evaluator aligns on shared protein `CA` residues and reports RMSD on shared nucleic-acid `C4'` atoms, so AF3 side-chain completion on the protein chain does not break scoring.

### Protein Binding Ligand (PBL)

PBL evaluates ligand-containing protein structures and reports geometry, chemistry, and optional Vina docking metrics.

- Input: `.cif` files in `design_dir`
- Accepted layout: either `design_dir/*.cif` or nested case folders such as `design_dir/2vt4/*.cif`
- Evaluation: automatic ligand extraction, pocket extraction, ligand geometry metrics, chemistry metrics, and Vina docking metrics

If your input CIF files are already inverse-fold outputs, the current PBL pipeline skips the inverse-fold stage and evaluates them directly after preprocessing.

Minimal run command:

```bash
python scripts/run_pbl_pipeline.py \
  design_dir=/path/to/protein_binding_ligand_designs \
  gpus=0
```

Optional: set `root=/path/to/output_dir` to change output location (default: `results`).

Recommended example layout:

```text
examples/protein_binding_ligand/
`-- 2vt4/
    `-- 2vt4-1_seed_42_bb_0_seq_0.cif
```

Example run:

```bash
python scripts/run_pbl_pipeline.py \
  design_dir=examples/protein_binding_ligand \
  gpus=0 \
  root=results/examples/protein_binding_ligand
```

Successful runs will write:

- preprocessed CIFs to `formatted_designs/`
- evaluation inputs to `inversefold_formatted_designs_for_evaluation/`
- ligand metrics to `inversefold_formatted_designs_for_evaluation_metrics/`

The final CSV and summary JSON are:

- `inversefold_formatted_designs_for_evaluation_metrics/evaluation_results.csv`
- `inversefold_formatted_designs_for_evaluation_metrics/evaluation_summary_metrics.json`

### MotifBench (Motif Scaffolding)

MotifBench evaluates whether generated scaffolds preserve motif geometry while remaining structurally plausible and diverse.

- Input: scaffold PDB files + `scaffold_info.csv`
- Inverse fold: ProteinMPNN (motif-constrained design)
- Refold: ESMFold
- Evaluation: motif RMSD/scaffold RMSD, novelty, diversity

#### Foldseek Installation (Required for MotifBench Evaluation)

MotifBench uses Foldseek for diversity clustering and novelty evaluation. Follow these steps to install:

**1. Install Foldseek via conda:**

```bash
conda install -c conda-forge -c bioconda foldseek
```

**2. Download the Foldseek PDB database:**

```bash
export FOLDSEEK_DATABASE=/path/to/foldseek_pdb_database
mkdir -p $FOLDSEEK_DATABASE
cd $FOLDSEEK_DATABASE
foldseek databases PDB pdb tmp
```

> Note: The database download may take 30-60 minutes depending on your connection speed. The PDB database is approximately 60GB uncompressed.

**3. Set environment variables:**

```bash
# Add to your ~/.bashrc or ~/.zshrc for persistence
export FOLDSEEK_DATABASE=/path/to/foldseek_pdb_database
export FOLDSEEK_BIN=$(which foldseek)  # or explicitly: /path/to/conda/bin/foldseek
```

**4. Verify installation:**

```bash
foldseek --version
foldseek createdb --help
```

**5. Running MotifBench evaluation with Foldseek:**

```bash
python scripts/run_motif_scaffolding_pipeline.py \
  design_dir=/path/to/motif_scaffolds \
  gpus=0 \
  motif_scaffolding.foldseek_database=$FOLDSEEK_DATABASE/pdb
```

Or via environment variable:

```bash
export FOLDSEEK_DATABASE=/path/to/foldseek_pdb_database
python scripts/run_motif_scaffolding_pipeline.py \
  design_dir=/path/to/motif_scaffolds \
  gpus=0
```

Minimal run command:

```bash
python scripts/run_motif_scaffolding_pipeline.py \
  design_dir=/path/to/motif_scaffolds \
  gpus=0
```

Optional: set `root=/path/to/output_dir` to change output location (default: `results`).

`scaffold_info.csv` should include:

- `sample_num`
- `motif_placements`

Example:

```csv
sample_num,motif_placements
0,34/A/70
1,30/A/25/B/30
```

## Examples

We provide ready-to-use examples in the `examples/` directory. You can run them directly to verify your installation and understand the pipeline workflow.

### 1. Motif Scaffolding

```bash
python3 scripts/run_motif_scaffolding_pipeline.py design_dir=examples/motif_scaffolding/01_1LDB/ gpus=0 root=results/examples/motif_scaffolding
```

### 2. Protein Binding Protein (PBP)

```bash
python3 scripts/run_pbp_pipeline.py design_dir=examples/protein_binding_protein/ gpus=0 root=results/examples/protein_binding_protein
```

### 2. Ligand Binding Protein

```bash
cd /path/to/ODesignBench
python3 scripts/run_lbp_pipeline.py \
  design_dir=examples/ligand_binding_protein \
  gpus=0 \
  root=results/examples/ligand_binding_protein
```

If `preprocess/ccd_component/components.cif` is missing, download it first:

```bash
cd preprocess/ccd_component
wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
gunzip -f components.cif.gz
```

Example metadata:

```csv
design_name,target_chain,design_chain
FAD_seed_1_bb_0_seq_0,A,B
```

### 3. Interface Design

```bash
python3 scripts/run_interface_pipeline.py \
  design_dir=examples/ligand_binding_protein \
  gpus=0 \
  root=results/examples/interface_design
```

This task uses the same `examples/ligand_binding_protein/lbp_info.csv` metadata file as LBP.

### 4. Atomic Motif Scaffolding / Enzyme Design (AME)

```bash
python3 scripts/run_ame_pipeline.py design_dir=examples/tip_atom_scaffolding/ gpus=0 root=results/examples/tip_atom_scaffolding
```

### 5. Protein Binding Ligand (PBL)

```bash
python3 scripts/run_pbl_pipeline.py design_dir=examples/protein_binding_ligand/ gpus=0 root=results/examples/protein_binding_ligand
```

### 6. Nucleic Acid (RNA example)

```bash
python3 scripts/run_nuc_pipeline.py design_dir=examples/nuc/rna/ refold.run_data_pipeline=true gpus=0 root=results/examples/nuc
```

### 7. Protein Binding Nucleic Acid (PBN)

```bash
python3 scripts/run_pbn_pipeline.py \
  design_dir=examples/protein_binding_nuc \
  gpus=0 \
  root=results/examples/protein_binding_nuc
```

## Repository Layout

- `scripts/`: task-level pipeline entry points
- `configs/`: Hydra/OmegaConf configuration groups
- `preprocess/`: input standardization and conversion utilities
- `inversefold/`: sequence design backends
- `refold/`: structure prediction/refolding backends
- `evaluation/`: task-specific metrics and evaluators
- `assets/`: benchmark assets and reference metadata

## General Usage Pattern

Most tasks follow the same lifecycle:

1. Provide standardized input structures and task metadata.
2. Run inverse folding to generate sequences.
3. Refold generated sequences to structures.
4. Compute benchmark metrics and export CSV results.

### Running Specific Pipeline Steps

The unified pipeline consists of five main stages: `preprocess`, `inversefold`, `refold_prepare`, `refold`, and `evaluation`. By default, all stages run sequentially. 

You can skip specific stages by setting them to `false` via Hydra overrides using the `+unified.steps.<stage_name>=false` syntax. This is particularly useful if you only want to re-run evaluation or skip a time-consuming step that has already completed.

**Example: Skip preprocessing and inverse folding**

```bash
python scripts/run_ame_pipeline.py design_dir=examples/tip_atom_scaffolding/ gpus=0 root=results/examples/tip_atom_scaffolding \
  +unified.steps.preprocess=false \
  +unified.steps.inversefold=false
```

**Example: Run ONLY evaluation (skip all other steps)**

```bash
python scripts/run_ame_pipeline.py design_dir=examples/tip_atom_scaffolding/ gpus=0 root=results/examples/tip_atom_scaffolding \
  +unified.steps.preprocess=false \
  +unified.steps.inversefold=false \
  +unified.steps.refold_prepare=false \
  +unified.steps.refold=false
```

## License and Citation

Please cite the corresponding benchmark release and model/tool dependencies used in your run (for example, PyRosetta, ESM, Chai-1, and AlphaFold3 where applicable).
