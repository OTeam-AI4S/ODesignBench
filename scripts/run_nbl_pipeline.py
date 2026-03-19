import hydra
from omegaconf import DictConfig

from pipeline_framework import run_unified_pipeline

@hydra.main(config_path="../configs", config_name="config_nuc_binding_ligand")
def main(cfg: DictConfig):
    run_unified_pipeline(cfg, task_name="nbl")

if __name__ == "__main__":
    main()