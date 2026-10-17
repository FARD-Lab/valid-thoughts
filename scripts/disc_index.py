import hydra
from omegaconf import OmegaConf

from src.dataset.curator import DataDiscIndex
from src.utils import set_seed_all
from src.utils.config import DataDiscIndexConfig, register_configs
from src.utils.logging import Logger

register_configs()

@hydra.main(version_base=None, config_path="../configs", config_name="disc_index")
def main(cfg: DataDiscIndexConfig) -> None:
    print("=" * 20)
    print("Configuration:")
    print("=" * 20)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 20)

    cfg.logging.name = cfg.run_name

    logger = Logger.from_config(
        config=OmegaConf.to_container(cfg.logging, resolve=True),
    )

    logger.info("Starting Phase 2 - Split Generation")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Run name: {cfg.run_name or 'auto-generated'}")

    logger.info("Creating Phase 2 curator")
    curator = DataDiscIndex.from_config(cfg=cfg, logger=logger)

    logger.info("Running split generation")
    set_seed_all(cfg.seed)
    curator.run()

    logger.info("Phase 2 complete!")

if __name__ == "__main__":
    main()
