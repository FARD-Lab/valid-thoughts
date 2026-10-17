import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.dataset.curator import DataLLM
from src.model.llm import BaseLLM
from src.utils import set_seed_all
from src.utils.config import DataLLMConfig, register_configs
from src.utils.logging import Logger

register_configs()

@hydra.main(version_base=None, config_path="../configs", config_name="llm_data")
def main(cfg: DataLLMConfig) -> None:
    print("=" * 20)
    print("Configuration:")
    print("=" * 20)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 20)

    cfg.logging.name = cfg.run_name

    logger = Logger.from_config(
        config=OmegaConf.to_container(cfg.logging, resolve=True),
    )

    logger.info("Starting dataset curation")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Run name: {cfg.run_name or 'auto-generated'}")

    logger.info(f"Instantiating base LLM: {cfg.base_llm.name}")
    base_llm = BaseLLM.from_config(cfg.base_llm, logger)
    base_llm.validate_offline_cache()

    logger.info(f"Creating data loader: {cfg.loader._target_}")
    data_loader = instantiate(cfg.loader, logger=logger)

    logger.info("Creating dataset curator")
    curator = DataLLM.from_config(
        cfg=cfg, base_llm=base_llm, data_loader=data_loader, logger=logger
    )

    logger.info("Running dataset curation pipeline")
    set_seed_all(cfg.seed)
    curator.run()

    logger.info("Dataset curation complete!")

if __name__ == "__main__":
    main()
