import json
from pathlib import Path
from typing import Dict, Optional, Set

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.disc_loader import ExampleLoader
from src.model.embedder import Embedder
from src.model.llm import BaseLLM
from src.utils import set_seed_all
from src.utils.config import (
    DiscDataConfig,
    ThoughtRepresentation,
    register_configs,
)
from src.utils.logging import Logger

register_configs()

EMBEDDER_TR_TYPES = {
    ThoughtRepresentation.EMBEDDING_POOLING,
    ThoughtRepresentation.EMBEDDING_NO_POOLING,
    ThoughtRepresentation.EMBEDDING_ALL,
    ThoughtRepresentation.INPUT_EMBEDDING,
}
LLM_TR_TYPES = {
    ThoughtRepresentation.SOFT_THINKING,
    ThoughtRepresentation.SOFT_THINKING_NOISE,
    ThoughtRepresentation.LATENT_THINKING,
}
FILE_TR_TYPES = {
    ThoughtRepresentation.LAST_INPUT_TOKEN,
}

class DataPreGenerator:
    def __init__(self, config: DiscDataConfig, logger: Logger):
        self.config = config
        self.logger = logger

        self.loader = ExampleLoader(
            directory=self.config.llm_data_output_dir,
            num_return_sequences=self.config.num_return_sequences,
            shard_size=self.config.shard_size,
        )

        self.base_llm: Optional[BaseLLM] = None
        self.embedder: Optional[Embedder] = None

        self.disc_index_dir = Path(self.config.disc_index_output_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.think_steps = self.config.think_steps
        self.source_hidden_size = self.config.base_llm.hidden_size
        self.checkpoint_interval = 100

        if self.config.tr_types is None:
            self.tr_types_to_process = set(ThoughtRepresentation)
        else:
            self.tr_types_to_process = {
                ThoughtRepresentation(tr_type) for tr_type in self.config.tr_types
            }
        self.logger.info(
            f"Processing thought representations: {[tr.value for tr in self.tr_types_to_process]}"
        )

    def _compute_thought_vec(
        self, example: dict, tr_type: ThoughtRepresentation, add_noise: bool = True
    ) -> torch.Tensor:
        generated_texts = example["generated_texts"]

        if tr_type == ThoughtRepresentation.LAST_INPUT_TOKEN:
            first_hs_path = Path(example["first_hs_path"])
            if not first_hs_path.exists():
                raise FileNotFoundError(f"first_hs not found: {first_hs_path}")
            thought_vec = torch.load(first_hs_path, map_location="cpu")

        elif tr_type == ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE:
            raise ValueError(
                "LAST_INPUT_HIDDEN_STATE should share the same file as LAST_INPUT_TOKEN"
            )

        elif tr_type in [
            ThoughtRepresentation.EMBEDDING_POOLING,
            ThoughtRepresentation.EMBEDDING_NO_POOLING,
            ThoughtRepresentation.EMBEDDING_ALL,
        ]:
            with torch.no_grad():
                embeddings = self.embedder.embed(generated_texts).to(
                    dtype=torch.bfloat16
                )
            if tr_type == ThoughtRepresentation.EMBEDDING_POOLING:
                thought_vec = torch.nn.functional.normalize(
                    torch.mean(embeddings, dim=0, keepdim=True), p=2, dim=1
                )
            elif tr_type == ThoughtRepresentation.EMBEDDING_NO_POOLING:
                thought_vec = embeddings.unsqueeze(1)
            elif tr_type == ThoughtRepresentation.EMBEDDING_ALL:
                thought_vec = embeddings

        elif tr_type == ThoughtRepresentation.SOFT_THINKING:
            thought_vec = self.base_llm.get_soft_token(
                text=example["input_text"],
                instruction=example["instruction"],
                add_noise=add_noise,
                steps=self.think_steps,
            )[0]

        elif tr_type == ThoughtRepresentation.LATENT_THINKING:
            thought_vec = self.base_llm.get_latent_thinking(
                text=example["input_text"],
                instruction=example["instruction"],
                steps=self.think_steps,
            )[0]

        elif tr_type == ThoughtRepresentation.INPUT_EMBEDDING:
            input_text = example["input_text"]
            with torch.no_grad():
                thought_vec = self.embedder.embed([input_text]).to(
                    dtype=torch.bfloat16
                )

        elif tr_type == ThoughtRepresentation.RANDOM_VECTOR:
            thought_vec = torch.randn(
                (len(generated_texts), 1, self.source_hidden_size),
                dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unsupported representation: {tr_type.value}")

        return thought_vec

    def _get_checkpoint_path(self, split_name: str) -> Path:
        return self.output_dir / f"{split_name}_checkpoint.pt"

    def _save_checkpoint(
        self,
        split_name: str,
        processed_groups: set,
        generated_texts_cache: dict,
        thought_vecs_caches: dict,
    ):
        checkpoint_path = self._get_checkpoint_path(split_name)
        checkpoint = {
            "processed_groups": processed_groups,
            "generated_texts_cache": generated_texts_cache,
            "thought_vecs_caches": thought_vecs_caches,
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(
            f"Saved checkpoint with {len(processed_groups)} processed groups"
        )

    def _load_checkpoint(
        self, split_name: str, tr_types: Set[ThoughtRepresentation]
    ) -> tuple:
        checkpoint_path = self._get_checkpoint_path(split_name)
        if checkpoint_path.exists():
            self.logger.info(f"Found checkpoint at {checkpoint_path}, resuming...")
            checkpoint =torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            processed_groups = checkpoint["processed_groups"]
            generated_texts_cache = checkpoint["generated_texts_cache"]
            loaded_caches = checkpoint["thought_vecs_caches"]

            thought_vecs_caches = {
                tr_type: loaded_caches.get(tr_type, {}) for tr_type in tr_types
            }

            self.logger.info(
                f"Resuming from checkpoint with {len(processed_groups)} already processed groups"
            )
            return processed_groups, generated_texts_cache, thought_vecs_caches
        else:
            return set(), {}, {tr_type: {} for tr_type in tr_types}

    def _delete_checkpoint(self, split_name: str):
        checkpoint_path = self._get_checkpoint_path(split_name)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            self.logger.info(f"Deleted checkpoint file: {checkpoint_path}")

    def _get_groups_file(self, split_name: str) -> Path:
        return self.disc_index_dir / f"{split_name}_groups.json"

    def _load_groups(self, split_name: str) -> Set[int]:
        with open(self._get_groups_file(split_name), "r") as f:
            return set(json.load(f))

    def _process_split(
        self,
        split_name: str,
        tr_types: Set[ThoughtRepresentation],
        save_generated_texts: bool = True,
    ) -> tuple:
        groups_file = self._get_groups_file(split_name)

        self.logger.info(
            f"Processing {split_name} split from {groups_file} "
            f"for TR types: {[t.value for t in tr_types]}..."
        )

        unique_groups = self._load_groups(split_name)

        self.logger.info(f"Found {len(unique_groups)} unique groups")

        (
            processed_groups,
            generated_texts_cache,
            thought_vecs_caches,
        ) = self._load_checkpoint(split_name, tr_types)

        remaining_groups = [g for g in unique_groups if g not in processed_groups]
        self.logger.info(
            f"Processing {len(remaining_groups)} remaining groups "
            f"(already processed: {len(processed_groups)})"
        )

        groups_since_checkpoint = 0

        combinable_types = {
            ThoughtRepresentation.SOFT_THINKING,
            ThoughtRepresentation.SOFT_THINKING_NOISE,
            ThoughtRepresentation.LATENT_THINKING,
        }
        use_combined = combinable_types.issubset(tr_types) and self.base_llm is not None
        remaining_tr_types = tr_types - combinable_types if use_combined else tr_types

        if use_combined:
            self.logger.info(
                "Using combined thinking (shared prefill + batched decode) "
                "for soft_thinking, soft_thinking_noise, and latent_thinking"
            )

        self.logger.info(
            f"Loading groups and computing TR types for {split_name}..."
        )
        for group_idx in tqdm(
            remaining_groups, desc=f"Processing groups for {split_name}"
        ):
            try:
                example = self.loader.load_example(group_idx)
                generated_texts = example["generated_texts"]

                if save_generated_texts:
                    generated_texts_cache[group_idx] = generated_texts

                if use_combined:
                    soft, noise, latent = self.base_llm.get_combined_thinking(
                        text=example["input_text"],
                        instruction=example["instruction"],
                        steps=self.think_steps,
                    )
                    thought_vecs_caches[ThoughtRepresentation.SOFT_THINKING][group_idx] = soft[0]
                    thought_vecs_caches[ThoughtRepresentation.SOFT_THINKING_NOISE][group_idx] = noise[0]
                    thought_vecs_caches[ThoughtRepresentation.LATENT_THINKING][group_idx] = latent[0]

                for tr_type in remaining_tr_types:
                    if tr_type in [
                        ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
                        ThoughtRepresentation.RANDOM_VECTOR,
                    ]:
                        continue
                    elif tr_type == ThoughtRepresentation.SOFT_THINKING:
                        thought_vec = self._compute_thought_vec(
                            example, tr_type, add_noise=False
                        )
                        thought_vecs_caches[tr_type][group_idx] = thought_vec
                    elif tr_type == ThoughtRepresentation.SOFT_THINKING_NOISE:
                        thought_vec = self._compute_thought_vec(
                            example,
                            ThoughtRepresentation.SOFT_THINKING,
                            add_noise=True,
                        )
                        thought_vecs_caches[tr_type][group_idx] = thought_vec
                    else:
                        thought_vec = self._compute_thought_vec(example, tr_type)
                        thought_vecs_caches[tr_type][group_idx] = thought_vec

                processed_groups.add(group_idx)
                groups_since_checkpoint += 1

                if groups_since_checkpoint >= self.checkpoint_interval:
                    self._save_checkpoint(
                        split_name,
                        processed_groups,
                        generated_texts_cache,
                        thought_vecs_caches,
                    )
                    groups_since_checkpoint = 0

            except Exception as e:
                self.logger.warning(
                    f"Error processing group {group_idx}, saving checkpoint before exit..."
                )
                self._save_checkpoint(
                    split_name,
                    processed_groups,
                    generated_texts_cache,
                    thought_vecs_caches,
                )
                self.logger.exception(f"Error processing group {group_idx}: {e}")
                raise e

        if groups_since_checkpoint > 0:
            self._save_checkpoint(
                split_name,
                processed_groups,
                generated_texts_cache,
                thought_vecs_caches,
            )

        return generated_texts_cache, thought_vecs_caches

    def _save_split_outputs(
        self,
        split_name: str,
        generated_texts_cache: Dict,
        thought_vecs_caches: Dict,
        save_generated_texts: bool,
    ):
        if save_generated_texts and generated_texts_cache:
            gen_texts_file = self.output_dir / f"{split_name}_generated_texts.pt"
            self.logger.info(
                f"Saving generated_texts with {len(generated_texts_cache)} groups "
                f"to {gen_texts_file}..."
            )
            torch.save(generated_texts_cache, gen_texts_file)
            gen_texts_size_mb = gen_texts_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"Generated texts file size: {gen_texts_size_mb:.2f} MB")

        for tr_type, thought_vecs_cache in thought_vecs_caches.items():
            if not thought_vecs_cache:
                self.logger.info(f"Skipping empty cache for {tr_type.value}")
                continue

            vectors_file = (
                self.output_dir / f"{split_name}_other_vectors_{tr_type.value}.pt"
            )
            self.logger.info(
                f"Saving thought_vecs for {tr_type.value} with "
                f"{len(thought_vecs_cache)} groups to {vectors_file}..."
            )
            torch.save(thought_vecs_cache, vectors_file)
            vectors_size_mb = vectors_file.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"{tr_type.value} vectors file size: {vectors_size_mb:.2f} MB"
            )

    def _run_pass(
        self,
        tr_types: Set[ThoughtRepresentation],
        save_generated_texts: bool,
    ):
        for split_name in ["train", "val", "test"]:
            groups_file = self._get_groups_file(split_name)
            if not groups_file.exists():
                self.logger.warning(
                    f"Groups file not found: {groups_file}, skipping {split_name}"
                )
                continue

            generated_texts_cache, thought_vecs_caches = self._process_split(
                split_name, tr_types, save_generated_texts
            )

            self._save_split_outputs(
                split_name,
                generated_texts_cache,
                thought_vecs_caches,
                save_generated_texts,
            )

            self._delete_checkpoint(split_name)

            self.logger.info(f"Saved {split_name} split successfully")

    def _get_expected_groups(self, split_name: str) -> Optional[int]:
        if not self._get_groups_file(split_name).exists():
            return None
        return len(self._load_groups(split_name))

    def _pass_outputs_complete(
        self, tr_types: Set[ThoughtRepresentation], check_generated_texts: bool
    ) -> bool:
        splits = ["train", "val", "test"]
        for split_name in splits:
            expected = self._get_expected_groups(split_name)
            if expected is None:
                continue

            if check_generated_texts:
                gen_file = self.output_dir / f"{split_name}_generated_texts.pt"
                if not gen_file.exists():
                    return False
                data = torch.load(gen_file, map_location="cpu")
                if len(data) != expected:
                    self.logger.info(
                        f"{gen_file.name}: {len(data)}/{expected} groups, incomplete"
                    )
                    return False

            for tr_type in tr_types:
                if tr_type in [
                    ThoughtRepresentation.LAST_INPUT_HIDDEN_STATE,
                    ThoughtRepresentation.RANDOM_VECTOR,
                ]:
                    continue
                vec_file = (
                    self.output_dir
                    / f"{split_name}_other_vectors_{tr_type.value}.pt"
                )
                if not vec_file.exists():
                    return False
                data = torch.load(vec_file, map_location="cpu")
                if len(data) != expected:
                    self.logger.info(
                        f"{vec_file.name}: {len(data)}/{expected} groups, incomplete"
                    )
                    return False
        return True

    def run(self):
        embedder_types = self.tr_types_to_process & EMBEDDER_TR_TYPES
        llm_types = self.tr_types_to_process & LLM_TR_TYPES
        file_types = self.tr_types_to_process & FILE_TR_TYPES

        need_embedder = bool(embedder_types)

        if need_embedder:
            if self._pass_outputs_complete(embedder_types, check_generated_texts=True):
                self.logger.info(
                    "=== Pass 1: Embedder phase === SKIPPED (outputs already exist)"
                )
            else:
                self.logger.info(
                    "=== Pass 1: Embedder phase ==="
                    f" TR types: {[t.value for t in embedder_types]}"
                )
                self.embedder = Embedder.from_config(
                    config=self.config.embedder,
                    logger=self.logger,
                )
                self.embedder.load()
                self.embedder.eval()

                self._run_pass(embedder_types, save_generated_texts=True)

                del self.embedder
                self.embedder = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Embedder unloaded, CUDA cache cleared.")
        else:
            self.logger.info("No embedding TR types requested, skipping embedder pass.")

        llm_pass_types = llm_types | file_types
        if llm_pass_types:
            if self._pass_outputs_complete(
                llm_pass_types, check_generated_texts=not need_embedder
            ):
                self.logger.info(
                    "=== Pass 2: LLM phase === SKIPPED (outputs already exist)"
                )
            else:
                self.logger.info(
                    "=== Pass 2: LLM phase ==="
                    f" TR types: {[t.value for t in llm_pass_types]}"
                )
                self.base_llm = BaseLLM.from_config(
                    config=self.config.base_llm,
                    logger=self.logger,
                )
                self.base_llm.load()
                self.base_llm.eval()

                self._run_pass(llm_pass_types, save_generated_texts=not need_embedder)

                del self.base_llm
                self.base_llm = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("LLM unloaded, CUDA cache cleared.")
        else:
            self.logger.info("No LLM TR types requested, skipping LLM pass.")

        self.logger.info("Pre-generation completed successfully!")

@hydra.main(version_base=None, config_path="../configs", config_name="disc_data")
def main(cfg: DiscDataConfig) -> None:
    logger = Logger.from_config(cfg.logging)

    logger.info("Pre-generation Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        set_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    generator = DataPreGenerator(config=cfg, logger=logger)
    generator.run()

if __name__ == "__main__":
    main()
