import hydra
from omegaconf import DictConfig

from pipeline_framework import run_unified_pipeline


@hydra.main(config_path="../configs", config_name="config_atomic_motif_enzyme", version_base=None)
def main(cfg: DictConfig):
    run_unified_pipeline(cfg, task_name="ame")


if __name__ == "__main__":
    main()
