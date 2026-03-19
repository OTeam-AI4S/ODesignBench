import hydra
from omegaconf import DictConfig

from pipeline_framework import run_unified_pipeline

@hydra.main(config_path="../configs", config_name="config_protein")
def main(cfg: DictConfig):
    run_unified_pipeline(cfg, task_name="protein")

if __name__ == "__main__":
    main()