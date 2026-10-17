import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.dataset.base import BaseLoader
from src.utils.config import BBEHLoaderConfig
from src.utils.logging import Logger

class BBEHLoader(BaseLoader):
    INSTRUCTION = (
        'Think step by step, and when you provide the final answer, please use the prefix "The answer is:"\
without any modification, and provide the answer directly, with no formatting, no bolding, and\
no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple\
choice with a single correct answer, the final answer must only be the letter corresponding to\
the correct answer. For example, "The answer is: (a)".'
    )

    TASK_NAMES = [
        "bbeh_boardgame_qa",
        "bbeh_boolean_expressions",
        "bbeh_buggy_tables",
        "bbeh_causal_understanding",
        "bbeh_disambiguation_qa",
        "bbeh_dyck_languages",
        "bbeh_geometric_shapes",
        "bbeh_hyperbaton",
        "bbeh_linguini",
        "bbeh_movie_recommendation",
        "bbeh_multistep_arithmetic",
        "bbeh_nycc",
        "bbeh_object_counting",
        "bbeh_object_properties",
        "bbeh_sarc_triples",
        "bbeh_shuffled_objects",
        "bbeh_spatial_reasoning",
        "bbeh_sportqa",
        "bbeh_temporal_sequence",
        "bbeh_time_arithmetic",
        "bbeh_web_of_lies",
        "bbeh_word_sorting",
        "bbeh_zebra_puzzles",
    ]

    def __init__(
        self,
        logger: Logger,
        data_dir: str,
        task_name: Optional[str] = None,
        num_examples: Optional[int] = None,
        example_start_idx: Optional[int] = None,
        example_end_idx: Optional[int] = None,
    ):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.num_examples = num_examples
        self.example_start_idx = example_start_idx
        self.example_end_idx = example_end_idx
        self.tasks = {}

        if task_name and task_name not in self.TASK_NAMES:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {', '.join(self.TASK_NAMES)}"
            )

    @staticmethod
    def from_config(config: BBEHLoaderConfig, logger: Logger) -> "BBEHLoader":
        return BBEHLoader(
            logger=logger,
            data_dir=config.data_dir,
            task_name=config.task_name,
            num_examples=config.num_examples,
            example_start_idx=config.example_start_idx,
            example_end_idx=config.example_end_idx,
        )

    def load(self) -> None:
        tasks_to_load = [self.task_name] if self.task_name else self.TASK_NAMES

        for task_name in tasks_to_load:
            task_dir = self.data_dir / task_name
            task_file = task_dir / "task.json"

            if not task_file.exists():
                self.logger.warning(
                    f"Task file not found: {task_file}. Skipping {task_name}."
                )
                continue

            self.logger.info(f"Loading {task_name} from {task_file}")

            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                examples = task_data.get("examples", [])
                canary = task_data.get("canary", "")

            self.tasks[task_name] = {
                "examples": examples,
                "canary": canary,
            }

            self.logger.info(f"Loaded {len(examples)} examples from {task_name}")

        if not self.tasks:
            raise RuntimeError(
                f"No tasks loaded. Check that task directories exist in {self.data_dir}"
            )

        self.logger.info(
            f"Successfully loaded {len(self.tasks)} task(s): "
            f"{', '.join(self.tasks.keys())}"
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self.tasks:
            raise RuntimeError("No tasks loaded. Call load() first.")

        self.count = 0
        start = self.example_start_idx
        end = self.example_end_idx
        for task_name, task_data in self.tasks.items():
            examples = task_data["examples"]
            canary = task_data["canary"]

            for pos, example in enumerate(examples):
                if start is not None and pos < start:
                    continue
                if end is not None and pos >= end:
                    break
                if self.num_examples is not None and self.count >= self.num_examples:
                    return

                example_with_metadata = {
                    "task_name": task_name,
                    "input": example.get("input", ""),
                    "target": example.get("target", ""),
                    "canary": canary,
                }
                yield example_with_metadata
                self.count += 1

    def get_text_field(self, example: Dict[str, Any]) -> str:
        return example.get("input", "")

    def reject_example(self, example: Dict[str, Any]):
        self.count -= 1

    @property
    def instruction(self) -> str:
        return BBEHLoader.INSTRUCTION

