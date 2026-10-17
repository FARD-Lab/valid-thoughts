import gc
import json
import random
import time
from pathlib import Path

import torch

from src.dataset.base import BaseLoader
from src.model.llm import BaseLLM, len_output_sequence
from src.utils import get_subdirs_llm_data
from src.utils.config import DataDiscIndexConfig, DataLLMConfig
from src.utils.logging import Logger

class DataLLM:
    def __init__(
        self,
        logger: Logger,
        base_llm: BaseLLM,
        data_loader: BaseLoader,
        output_dir: str,
        max_input_length: int,
        max_new_tokens: int,
        batch_size: int,
        num_return_sequences: int,
        return_hidden_states: bool,
        return_logits: bool,
        save_format: str,
        shard_size: int,
    ):
        self.base_llm = base_llm
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.num_return_sequences = num_return_sequences
        self.return_hidden_states = return_hidden_states
        self.return_logits = return_logits
        self.save_format = save_format
        self.shard_size = shard_size
        self.logger = logger

    @staticmethod
    def from_config(
        cfg: DataLLMConfig,
        base_llm: BaseLLM,
        data_loader: BaseLoader,
        logger: Logger,
    ) -> "DataLLM":
        return DataLLM(
            logger=logger,
            base_llm=base_llm,
            data_loader=data_loader,
            output_dir=cfg.output_dir,
            max_input_length=cfg.max_input_length,
            max_new_tokens=cfg.base_llm.max_new_tokens,
            batch_size=cfg.batch_size,
            num_return_sequences=cfg.base_llm.num_return_sequences or 1,
            return_hidden_states=cfg.base_llm.return_hidden_states,
            return_logits=cfg.base_llm.return_logits,
            save_format=cfg.save_format,
            shard_size=cfg.shard_size,
        )

    def run(self):
        self.logger.info("Starting dataset curation process")

        tensor_dir = Path(self.output_dir) / "tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.inputs_file = Path(self.output_dir) / "inputs.jsonl"

        n_done = 0
        shard_files = sorted(Path(self.output_dir).glob("generations_shard_*.jsonl"))
        if shard_files and self.inputs_file.exists():
            last_group_idx = -1
            for sf in shard_files:
                with open(sf, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            last_group_idx = max(last_group_idx, json.loads(line)["group_idx"])
            if last_group_idx >= 0:
                n_done = last_group_idx + 1
                with open(self.inputs_file, encoding="utf-8") as f:
                    kept = f.readlines()[:n_done]
                with open(self.inputs_file, "w", encoding="utf-8") as f:
                    f.writelines(kept)
                self.logger.info(
                    f"Resuming from group_idx={n_done} ({n_done} groups already done)"
                )

        self.data_loader.load()
        self.logger.info("Dataset loaded successfully")

        if shard_files and n_done > 0:
            self.current_shard_idx = len(shard_files) - 1
            with open(shard_files[-1], encoding="utf-8") as f:
                self.examples_in_current_shard = sum(1 for line in f if line.strip())
        else:
            self.current_shard_idx = 0
            self.examples_in_current_shard = 0

        def cleanup_memory():
            gc.collect()
            torch.cuda.empty_cache()

        self._run_single_pass(n_done, tensor_dir, cleanup_memory)

    def _run_single_pass(self, n_done: int, tensor_dir, cleanup_memory):
        self.base_llm.load()
        self.base_llm.eval()

        curr_batch_examples = []
        valid_examples = []
        global_idx = n_done

        with torch.no_grad():
            for i, example in enumerate(self.data_loader):
                if i < n_done:
                    continue
                curr_batch_examples.append(example)
                if len(curr_batch_examples) < self.batch_size:
                    continue

                tokenized = self.base_llm.tokenize(
                    [self.data_loader.get_text_field(ex) for ex in curr_batch_examples],
                    padding=False,
                    return_tensors=None,
                )
                for j in range(len(curr_batch_examples)):
                    input_ids = tokenized["input_ids"][j]
                    if len(input_ids) <= self.max_input_length:
                        valid_examples.append(curr_batch_examples[j])
                    else:
                        self.logger.info(
                            f"Rejecting example with length {len(input_ids)}"
                        )
                        self.data_loader.reject_example(curr_batch_examples[j])

                curr_batch_examples = []
                if len(valid_examples) < self.batch_size:
                    continue

                self.logger.info(f"Processing batch with Global idx: {global_idx}")
                batch_to_process = [
                    self.data_loader.get_text_field(ex)
                    for ex in valid_examples[: self.batch_size]
                ]
                valid_examples = valid_examples[self.batch_size :]

                batch_inputs_metadata = []
                for j, text in enumerate(batch_to_process):
                    batch_inputs_metadata.append({
                        "group_idx": global_idx + j,
                        "input_text": text,
                    })
                self._save_inputs(batch_inputs_metadata)

                gen_start = time.time()
                outputs = self.base_llm.generate(
                    batch_to_process,
                    instruction=self.data_loader.instruction,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.num_return_sequences,
                    return_hidden_states=self.return_hidden_states,
                    return_logits=self.return_logits,
                    early_stopping=True,
                    num_beams=self.num_return_sequences,
                )
                self.logger.info(
                    f"Generation completed in {time.time() - gen_start:.2f} seconds"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                lengths = len_output_sequence(
                    generated_ids=outputs["generated_ids"],
                    eos_token_id=self.base_llm.tokenizer.eos_token_id,
                    num_return_sequences=self.num_return_sequences,
                )

                self.logger.info("Extracting hidden states and logits...")
                generated_ids_cpu = outputs["generated_ids"].cpu()
                batch_generations_buffer = []
                batch_first_hs_buffer = []

                for j in range(self.batch_size):
                    current_group_idx = global_idx

                    first_hs_arr = None
                    if "hidden_states" in outputs and self.return_hidden_states:
                        idx_first = j * self.num_return_sequences
                        n_layers = len(outputs["hidden_states"][0])
                        first_hs_arr = torch.stack(
                            [
                                outputs["hidden_states"][0][layer_idx][
                                    idx_first, -1, :
                                ].cpu()
                                for layer_idx in range(n_layers)
                            ]
                        )

                    batch_first_hs_buffer.append({
                        "group_idx": current_group_idx,
                        "first_step_hidden_states": first_hs_arr,
                    })

                    for k in range(self.num_return_sequences):
                        idx = j * self.num_return_sequences + k
                        length = lengths[idx]

                        if "logits" in outputs and self.return_logits:
                            seq_logits = (
                                torch.stack(
                                    [step[idx] for step in outputs["logits"][:length]]
                                )
                                .float()
                                .cpu()
                            )
                        else:
                            seq_logits = None

                        rest_hs_arr = None

                        unique_id = current_group_idx * self.num_return_sequences + k
                        batch_generations_buffer.append({
                            "idx": unique_id,
                            "group_idx": current_group_idx,
                            "generated_text": self.base_llm.tokenizer.decode(
                                generated_ids_cpu[idx][:length],
                                skip_special_tokens=True,
                            ),
                            "rest_hidden_states": rest_hs_arr,
                            "logits": seq_logits,
                        })

                    global_idx += 1

                self.logger.info("Flushing batch generations to disk...")
                self._append_generations(
                    batch_generations_buffer, batch_first_hs_buffer, tensor_dir
                )

                del batch_generations_buffer
                del batch_first_hs_buffer
                del outputs
                del lengths
                cleanup_memory()

    def _save_inputs(self, inputs: list):
        mode = "a" if self.inputs_file.exists() else "w"
        with open(self.inputs_file, mode, encoding="utf-8") as f:
            for item in inputs:
                f.write(json.dumps(item) + "\n")

    def _append_generations(
        self, examples: list, first_hs_list: list, tensor_dir: Path
    ):
        if self.examples_in_current_shard >= self.shard_size:
            self.current_shard_idx += 1
            self.examples_in_current_shard = 0

        shard_filename = (
            f"generations_shard_{self.current_shard_idx}.{self.save_format}"
        )
        shard_path = Path(self.output_dir) / shard_filename

        metadata_to_save = []
        logits_dir = tensor_dir / "logits"
        first_hs_dir = tensor_dir / "first_hs"
        rest_hs_dir = tensor_dir / "rest_hs"
        logits_dir.mkdir(parents=True, exist_ok=True)
        first_hs_dir.mkdir(parents=True, exist_ok=True)
        rest_hs_dir.mkdir(parents=True, exist_ok=True)

        first_hs_path_map = {}
        for first_hs_entry in first_hs_list:
            group_idx = first_hs_entry["group_idx"]
            first_hs_arr = first_hs_entry["first_step_hidden_states"]

            first_hs_path = ""
            if first_hs_arr is not None and isinstance(first_hs_arr, torch.Tensor):
                p = first_hs_dir / f"first_hs_group_{group_idx}.pt"
                torch.save(first_hs_arr, p)
                first_hs_path = str(p)

            first_hs_path_map[group_idx] = first_hs_path

        for ex in examples:
            unique_id = ex["idx"]
            group_idx = ex["group_idx"]

            logits_path = ""
            if ex["logits"] is not None and isinstance(ex["logits"], torch.Tensor):
                p = logits_dir / f"logits_{unique_id}.pt"
                torch.save(ex["logits"], p)
                logits_path = str(p)

            rest_hs_path = ""
            if ex["rest_hidden_states"] is not None and isinstance(
                ex["rest_hidden_states"], torch.Tensor
            ):
                p = rest_hs_dir / f"rest_hs_{unique_id}.pt"
                torch.save(ex["rest_hidden_states"], p)
                rest_hs_path = str(p)

            metadata_to_save.append(
                {
                    "idx": unique_id,
                    "group_idx": group_idx,
                    "generated_text": ex["generated_text"],
                    "logits_file": logits_path,
                    "first_hs_file": first_hs_path_map[group_idx],
                    "rest_hs_file": rest_hs_path,
                }
            )

        if self.save_format == "jsonl":
            mode = "a" if shard_path.exists() else "w"
            with open(shard_path, mode, encoding="utf-8") as f:
                for meta in metadata_to_save:
                    f.write(json.dumps(meta) + "\n")
            self.examples_in_current_shard += len(metadata_to_save)

        elif self.save_format == "parquet":
            import pandas as pd

            batch_filename = f"generations_shard_{self.current_shard_idx}_batch_{int(time.time())}.parquet"
            batch_path = Path(self.output_dir) / batch_filename
            pd.DataFrame(metadata_to_save).to_parquet(batch_path)
            self.examples_in_current_shard += len(metadata_to_save)

class DataDiscIndex:
    def __init__(
        self,
        logger: Logger,
        num_return_sequences: int,
        total_rows: int,
        llm_data_dir: str,
        output_dir: str,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.num_return_sequences = num_return_sequences
        self.total_rows = total_rows
        self.llm_data_dir = llm_data_dir
        self.output_dir = output_dir
        self.logger = logger
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.rng = random.Random(self.seed)

    @staticmethod
    def from_config(
        cfg: DataDiscIndexConfig,
        logger: Logger,
    ) -> "DataDiscIndex":
        return DataDiscIndex(
            logger=logger,
            num_return_sequences=cfg.num_return_sequences,
            total_rows=cfg.total_rows,
            llm_data_dir=cfg.llm_data_output_dir,
            output_dir=cfg.output_dir,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            seed=cfg.seed,
        )

    def run(self):
        self.logger.info("Starting DataDiscIndex - Creating Stratified Task Splits")
        dirs = get_subdirs_llm_data(self.llm_data_dir)
        self.group_sizes = []
        for d in dirs:
            input_file = d / "inputs.jsonl"

            if input_file.is_file():
                with open(input_file, "r") as f:
                    line_count = sum(1 for _ in f)
                self.group_sizes.append(line_count)

        if self.total_rows % self.num_return_sequences != 0:
            raise ValueError(
                f"total_rows ({self.total_rows}) must be divisible by "
                f"num_return_sequences ({self.num_return_sequences})"
            )

        total_groups_from_sizes = sum(self.group_sizes)
        calculated_total_groups = self.total_rows // self.num_return_sequences

        if total_groups_from_sizes != calculated_total_groups:
            raise ValueError(
                f"Sum of group_sizes ({total_groups_from_sizes}) does not match "
                f"total groups calculated from rows ({calculated_total_groups})"
            )

        min_groups = min(self.group_sizes)
        min_ratio = min(self.train_ratio, self.val_ratio, self.test_ratio)
        mn_ass = int(min_groups * min_ratio)

        if mn_ass < 1:
            raise ValueError(
                "Constraint Failed: Each split must have at least 1 group per task, "
                f"but the smallest split has only {mn_ass} groups.\n"
                f"Detail: Min Task Size={min_groups}, Min Ratio={min_ratio}.\n"
                f"Please increase dataset size or the split ratios."
            )

        self.logger.info("Configuration:")
        self.logger.info(f"  Total rows: {self.total_rows}")
        self.logger.info(f"  Total groups: {calculated_total_groups}")
        self.logger.info(f"  Total topics: {len(self.group_sizes)}")

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits_structure = self._create_stratified_splits()

        for split_name in ["train", "validation", "test"]:
            topic_map = splits_structure[split_name]
            group_ids = [g for groups in topic_map.values() for g in groups]

            self.logger.info(f"  Split {split_name}: {len(group_ids)} groups")

            self._save_groups(group_ids, split_name, output_path)

        self.logger.info("Split generation completed successfully!")

    def _create_stratified_splits(self) -> dict:
        splits_structure = {"train": {}, "validation": {}, "test": {}}
        current_global_group_idx = 0

        for topic_idx, size in enumerate(self.group_sizes):

            topic_global_indices = list(
                range(current_global_group_idx, current_global_group_idx + size)
            )
            current_global_group_idx += size

            self.rng.shuffle(topic_global_indices)

            n_train = int(size * self.train_ratio)
            n_val = int(size * self.val_ratio)

            splits_structure["train"][topic_idx] = topic_global_indices[:n_train]
            splits_structure["validation"][topic_idx] = topic_global_indices[
                n_train : n_train + n_val
            ]
            splits_structure["test"][topic_idx] = topic_global_indices[
                n_train + n_val :
            ]

        return splits_structure

    def _save_groups(self, group_ids: list, split_name: str, output_path: Path):
        file_prefix = split_name if split_name != "validation" else "val"
        output_file = output_path / f"{file_prefix}_groups.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sorted(group_ids), f)

        self.logger.info(f"  Saved {len(group_ids)} groups to {output_file}")
