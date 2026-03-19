# Motif Scaffolding Module

This module integrates [MotifBench](https://github.com/trippelab/MotifBench) into designbench for evaluating motif scaffolding designs.

## Overview

Motif scaffolding is a central task in computational protein design: given the coordinates of atoms in a geometry chosen to confer a desired biochemical function (a motif), the goal is to identify diverse protein structures (scaffolds) that include the motif and stabilize its geometry.

This module integrates MotifBench evaluation into designbench following the **inversefold + refold paradigm**:

1. **InverseFold**: Uses designbench's InverseFold API (e.g., ProteinMPNN) to design sequences for given scaffolds
2. **ReFold**: Uses designbench's ReFold API (e.g., ESMFold) to predict structures from designed sequences
3. **MotifBench Analysis**: Calculates motif-specific metrics (RMSD, novelty, diversity) using MotifBench's analysis modules

## Requirements

1. **Motif PDB Files**: Motif PDB files must be available (see Configuration section)
2. **Foldseek**: Required for novelty calculation
3. **Python environment**: With required dependencies installed

## Configuration

Edit `configs/motif_scaffolding.yaml`:

```yaml
# Optional: Path to motif PDB files directory (defaults to internal resources/motif_pdbs)
motif_pdbs_dir: /path/to/motif_pdbs

python_path: /path/to/python
foldseek_database: /path/to/foldseek/database
gpu_id: 0  # Optional
motif_list: null  # Optional, evaluates all if null
```

**Note**: If `motif_pdbs_dir` is not specified, the system will use the default internal location at `evaluation/motif_scaffolding/resources/motif_pdbs/`. You can copy motif PDB files from MotifBench to this directory, or configure a custom path.

## Usage

### Basic Usage

```bash
python scripts/run_motif_scaffolding_pipeline.py \
    design_dir=/path/to/design/outputs \
    root=results
```

### Evaluate Specific Motifs

```bash
python scripts/run_motif_scaffolding_pipeline.py \
    design_dir=/path/to/design/outputs \
    motif_scaffolding.motif_list=[01_1LDB,02_1ITU,03_2CGA]
```

### Programmatic Usage

```python
from evaluation.motif_scaffolding.motif_scaffolding_evaluation import MotifScaffoldingEvaluation
from omegaconf import DictConfig

config = DictConfig({
    'motif_scaffolding': {
        'motif_pdbs_dir': '/path/to/motif_pdbs',  # Optional
        'python_path': '/path/to/python',
        'foldseek_database': '/path/to/foldseek/db'
    }
})

evaluation = MotifScaffoldingEvaluation(config)
results = api.run_motif_scaffolding_evaluation(
    design_dir='/path/to/designs',
    pipeline_dir='/path/to/results',
    motif_list=['01_1LDB', '02_1ITU']
)
```

## Input Format

The design directory should have the following structure:

```
design_dir/
├── 01_1LDB/
│   ├── scaffold_info.csv  # Required: contains sample_num and motif_placements
│   ├── 01_1LDB_0.pdb
│   ├── 01_1LDB_1.pdb
│   └── ...
├── 02_1ITU/
│   └── ...
└── ...
```

### scaffold_info.csv Format

```csv
sample_num,motif_placements
0,100/A/100
1,95/A/105
2,102/A/98
...
```

Where `motif_placements` follows MotifBench format: `{scaffold_len}/{motif_chain}/{scaffold_len}`

## Output

The pipeline generates evaluation results in:

```
pipeline_dir/
└── motif_scaffolding/
    ├── 01_1LDB/
    │   ├── scaffold_info.csv
    │   ├── motif_info.csv
    │   └── evaluation_results/
    │       ├── esm_complete_results.csv
    │       ├── esm_novelty_results.csv
    │       └── ...
    └── ...
```

## Evaluation Metrics

MotifBench evaluates designs on:

- **Success Rate**: Percentage of designs that successfully fold with the motif
- **RMSD**: Root mean square deviation of the refolded structure
- **TM-score**: Template modeling score
- **pLDDT**: Predicted LDDT (confidence score)
- **Novelty**: Structural novelty compared to PDB database
- **Diversity**: Structural diversity among successful designs

## Integration with designbench

This module can be integrated into the full designbench pipeline:

```python
# In a custom pipeline script
from preprocess.preprocess import Preprocess
from inversefold.inversefold_api import InverseFold
from refold.refold_api import ReFold
from motif_scaffolding import MotifScaffolding
from evaluation.evaluation_api import Evaluation

# ... run inversefold and refold ...

# Then evaluate with motif scaffolding
motif_evaluation = MotifScaffoldingEvaluation(config)
motif_evaluation.run_motif_scaffolding_evaluation(...)
```

## Notes

- The module assumes `scaffold_info.csv` already exists in the design directory
- For models that don't generate `scaffold_info.csv`, you'll need to implement a custom `prepare_scaffold_info` method
- MotifBench evaluation requires significant computational resources (GPU for ESMFold, CPU for foldseek)
