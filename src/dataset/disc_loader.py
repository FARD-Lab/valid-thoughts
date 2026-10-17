import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.dataset.bbeh_loader import BBEHLoader
from src.utils import get_subdirs_llm_data

class ExampleLoader:
    def __init__(
        self,
        directory: str = "./outputs/llm_data_8B/",
        num_return_sequences: int = 8,
        shard_size: int = 1024,
    ):
        self.directory = Path(directory)
        self.num_return_sequences = num_return_sequences
        self.shard_size = shard_size
        self.task_map = []
        self.total_examples = 0

        self._check_permissions()

        subdirs = get_subdirs_llm_data(self.directory)

        for sub in subdirs:
            instruction = None
            if sub.name.startswith("bbeh"):
                instruction = BBEHLoader.INSTRUCTION
            input_file = sub / "inputs.jsonl"
            if not input_file.exists():
                continue

            offsets = []
            with open(input_file, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets.append(offset)

            count = len(offsets)

            if count > 0:
                self.task_map.append(
                    {
                        "task": sub.name,
                        "path": sub,
                        "instruction": instruction,
                        "start": self.total_examples,
                        "end": self.total_examples + count,
                        "input_offsets": offsets,
                    }
                )
                self.total_examples += count

    def _check_permissions(self):
        if not self.directory.exists():
            raise PermissionError(f"Directory does not exist: {self.directory}")

        if not os.access(self.directory, os.R_OK):
            raise PermissionError(f"No read permission for directory: {self.directory}")

        permission_issues = []

        for subdir in self.directory.iterdir():
            if not subdir.is_dir():
                continue

            if not os.access(subdir, os.R_OK):
                permission_issues.append(f"No read permission: {subdir}")
                continue

            inputs_file = subdir / "inputs.jsonl"
            if inputs_file.exists() and not os.access(inputs_file, os.R_OK):
                permission_issues.append(f"No read permission: {inputs_file}")

            for shard_file in subdir.glob("generations_shard_*.jsonl"):
                if not os.access(shard_file, os.R_OK):
                    permission_issues.append(f"No read permission: {shard_file}")

        if permission_issues:
            raise PermissionError(
                "Permission issues found:\n"
                + "\n".join(f"  - {issue}" for issue in permission_issues)
            )

    def _read_generation_chunk(
        self, file_path: Path, start_line_idx: int, count: int
    ) -> List[dict]:
        results = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for _ in range(start_line_idx):
                    f.readline()
                for _ in range(count):
                    line = f.readline()
                    if not line:
                        raise EOFError(
                            "Reached end of file before reading enough lines"
                        )
                    results.append(json.loads(line))
        except FileNotFoundError:
            return []
        return results

    def load_example(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= self.total_examples:
            raise IndexError(f"Index {index} out of range")

        target_task = None
        local_group_idx = -1

        for meta in self.task_map:
            if meta["start"] <= index < meta["end"]:
                target_task = meta
                local_group_idx = index - meta["start"]
                break

        task_path = target_task["path"]

        input_data = ""
        offset = target_task["input_offsets"][local_group_idx]
        with open(task_path / "inputs.jsonl", "r", encoding="utf-8") as f:
            f.seek(offset)
            input_data = json.loads(f.readline()).get("input_text", "")

        start_gen_idx = local_group_idx * self.num_return_sequences

        shard_id = start_gen_idx // self.shard_size
        line_in_shard = start_gen_idx % self.shard_size
        shard_filename = f"generations_shard_{shard_id}.jsonl"
        shard_path = task_path / shard_filename

        raw_generations = self._read_generation_chunk(
            shard_path, line_in_shard, self.num_return_sequences
        )

        assert all(
            raw_generations[0].get("group_idx") == gen.get("group_idx")
            for gen in raw_generations
        ), "Mismatched group indices in loaded generations"

        generations = []
        rest_hs_paths = []

        for i in range(self.num_return_sequences):
            generations.append(raw_generations[i].get("generated_text"))
            rest_hs_paths.append(raw_generations[i].get("rest_hs_file"))

        first_hs_path = raw_generations[0]["first_hs_file"]

        return {
            "global_index": index,
            "task_name": target_task["task"],
            "instruction": target_task["instruction"],
            "group_idx": local_group_idx,
            "input_text": input_data,
            "generated_texts": generations,
            "first_hs_path": first_hs_path,
            "rest_hs_paths": rest_hs_paths,
        }

    def get_num_examples(self) -> int:
        return self.total_examples

if __name__ == "__main__":
    loader = ExampleLoader(
        directory="./outputs/llm_data_8B/",
        num_return_sequences=16,
        shard_size=1024,
    )

    total = loader.get_num_examples()
    print(f"Total examples indexed: {total}")
    sample_idx = 10
    example = loader.load_example(sample_idx)
    print(f"Example at index {sample_idx}:")
    print(example["input_text"])
