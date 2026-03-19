import os
import hydra
from omegaconf import DictConfig

from pipeline_framework import run_unified_pipeline

@hydra.main(config_path="../configs", config_name="config_protein_binding_protein")
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
    run_unified_pipeline(cfg, task_name="pbp")

if __name__ == "__main__":
    main()