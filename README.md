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

2. Create the Conda environment:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate designbench
```

### Notes

- `fair-esm==2.0.0` is used and is compatible with modern Python 3.10 and PyTorch 2.x.
- `openbabel` is included in `environment.yml` for ligand reconstruction/docking metrics (`mol_rec`, `rdkit_utils`, `docking_vina`).
- If your system has multiple CUDA installations, verify runtime visibility with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.

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

You can download them using the huggingface python module:

```bash
# Make sure you are in the ODesignBench directory
mkdir -p refold/esmfold/weights
python -m huggingface_hub.cli.hf download facebook/esmfold_v1 --local-dir refold/esmfold/weights
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

Some benchmark tasks use `refold=af3` which runs AlphaFold3 inside the Docker wrapper `refold/run_af3.sh`.

### Download required artifacts
Follow the official AlphaFold3 instructions in the upstream repo to download:
1. AF3 model parameters (see: [Obtaining model parameters](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters))
2. AF3 public databases (see the database download instructions in the upstream AlphaFold3 repo: https://github.com/google-deepmind/alphafold3).

### Directory layout expected by `refold/run_af3.sh`
The wrapper expects:
- `$AF3_BASE/model` -> mounted to `/root/models` inside the container
- `$AF3_PUBLIC_DB` -> mounted to `/root/public_databases` inside the container

### Configure environment variables
Set these before running any pipeline that uses AF3 refolding:
```bash
export AF3_BASE=/path/to/af3              # must contain: $AF3_BASE/model
export AF3_PUBLIC_DB=/path/to/public_databases
# Optional: if your docker image tag differs
export AF3_DOCKER_IMAGE=alphafold3
```

### Run
After exporting the variables above, run the normal pipeline commands (e.g. `scripts/run_pbp_pipeline.py` with `refold=af3`).

## Benchmark Content

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

### MotifBench (Motif Scaffolding)

MotifBench evaluates whether generated scaffolds preserve motif geometry while remaining structurally plausible and diverse.

- Input: scaffold PDB files + `scaffold_info.csv`
- Inverse fold: ProteinMPNN (motif-constrained design)
- Refold: ESMFold
- Evaluation: motif RMSD/scaffold RMSD, novelty, diversity

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

We provide ready-to-use examples for all three modules in the `examples/` directory. You can run them directly to verify your installation and understand the pipeline workflow.

### 1. Motif Scaffolding

```bash
python3 scripts/run_motif_scaffolding_pipeline.py design_dir=examples/motif_scaffolding/01_1LDB/ gpus=0 root=results/examples/motif_scaffolding
```

### 2. Protein Binding Protein (PBP)

```bash
python3 scripts/run_pbp_pipeline.py design_dir=examples/protein_binding_protein/ gpus=0 root=results/examples/protein_binding_protein
```

### 3. Atomic Motif Scaffolding / Enzyme Design (AME)

```bash
python3 scripts/run_ame_pipeline.py design_dir=examples/tip_atom_scaffolding/ gpus=0 root=results/examples/tip_atom_scaffolding
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
