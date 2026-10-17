#!/usr/bin/env python3

import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/google-deepmind/bbeh/80d12ca916b7158f22293fcf3144f4d3d854d4be/bbeh/benchmark_tasks"

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

def download_file(url: str, dest_path: Path, retry: int = 3) -> bool:
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")

            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()

            with open(dest_path, "wb") as f:
                f.write(content)

            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False

        except urllib.error.URLError:
            pass

        except Exception:
            pass

        if attempt < retry - 1:
            wait_time = 2**attempt
            time.sleep(wait_time)

    return False

def download_task(task_name: str, base_dir: Path) -> bool:
    task_dir = base_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {task_name}... ", end="", flush=True)

    task_json_url = f"{BASE_URL}/{task_name}/task.json"
    task_json_path = task_dir / "task.json"

    if not download_file(task_json_url, task_json_path):
        print("✗ (Failed to download task.json)")
        if task_dir.exists():
            shutil.rmtree(task_dir)
        return False

    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            example_count = len(data.get("examples", []))
    except (json.JSONDecodeError, Exception) as e:
        print(f"✗ (Invalid JSON: {e})")
        if task_dir.exists():
            shutil.rmtree(task_dir)
        return False

    readme_url = f"{BASE_URL}/{task_name}/README.md"
    readme_path = task_dir / "README.md"
    download_file(readme_url, readme_path)

    print(f"✓ ({example_count} examples)")
    return True

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    data_dir = project_root / "data" / "bbeh" / "benchmark_tasks"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BIG-Bench Extra Hard (BBEH) Dataset Downloader")
    print("=" * 80)
    print(f"\nDownloading to: {data_dir.absolute()}")
    print(f"Number of tasks: {len(TASK_NAMES)}\n")

    success_count = 0
    failed_tasks = []

    for task_name in TASK_NAMES:
        if download_task(task_name, data_dir):
            success_count += 1
        else:
            failed_tasks.append(task_name)

    print("\n" + "=" * 80)
    print(f"Download complete: {success_count}/{len(TASK_NAMES)} tasks")

    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for task_name in failed_tasks:
            print(f"  - {task_name}")
        print("\nYou can re-run this script to retry failed downloads.")
        sys.exit(1)
    else:
        print("\n✓ All tasks downloaded successfully!")

        total_examples = 0
        print("\nDataset statistics:")
        for task_name in sorted(TASK_NAMES):
            task_file = data_dir / task_name / "task.json"
            if task_file.exists():
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        count = len(data.get("examples", []))
                        total_examples += count
                        display_name = task_name.replace("bbeh_", "")
                        print(f"  {display_name:30s} {count:5d} examples")
                except Exception:
                    pass
        print(f"  {'TOTAL':30s} {total_examples:5d} examples")

    sys.exit(0)

if __name__ == "__main__":
    main()
