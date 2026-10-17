import argparse
import time
from pathlib import Path

import torch

from src.model.latent_methods import extract_answer, gold_answer, gsm8k, load_method
from src.utils.logging import Logger

METHODS = ["codi", "sim_cot", "colar", "llama", "codi_gpt2", "sim_cot_gpt2"]


def shard_name(split: str, start: int, end: int) -> str:
    return f"{split}_{start:05d}_{end:05d}"


def empty_record(method, split, samples, seed) -> dict:
    return dict(method=method.name, families=method.families, split=split, samples=samples, seed=seed,
                index=[], questions=[], gold=[], output_ids=[], outputs=[], explicit_correct=[],
                latent_answers=[], latent_correct=[], thoughts={family: [] for family in method.families},
                latent_by_family={family: [] for family in method.families},
                latent_correct_by_family={family: [] for family in method.families})


def add_latent(record: dict, family: str, vecs: torch.Tensor, answer, gold) -> None:
    record["thoughts"][family].append(vecs.float().cpu().clone())
    record["latent_by_family"][family].append(answer)
    record["latent_correct_by_family"][family].append(None if answer is None else extract_answer(answer) == gold)


def run(method, rows, split, start, samples, seed, path: Path, log, save_every: int = 10) -> dict:
    record = torch.load(path, weights_only=False) if path.exists() else empty_record(method, split, samples, seed)
    for i in range(len(record["questions"]), len(rows)):
        t0 = time.time()
        index = start + i
        question, gold = rows[i]["question"], gold_answer(rows[i]["answer"])
        with torch.no_grad():
            thoughts = method.thoughts(question, seed + index)
            answers = {f: method.latent_answer(question, thoughts[f], seed + index) for f in method.families}
            output_ids = method.sample_explicit(question, samples, seed + index)
        texts = [method.decode(ids) for ids in output_ids]
        answer = answers[method.families[0]]
        record["index"].append(index)
        record["questions"].append(question)
        record["gold"].append(gold)
        record["output_ids"].append(output_ids)
        record["outputs"].append(texts)
        record["explicit_correct"].append([extract_answer(t) == gold for t in texts])
        record["latent_answers"].append(answer)
        record["latent_correct"].append(None if answer is None else extract_answer(answer) == gold)
        for family, vecs in thoughts.items():
            add_latent(record, family, vecs, answers[family], gold)
        log(f"{method.name} {split} {index} latents {thoughts[method.families[0]].shape[0]} "
            f"explicit {sum(record['explicit_correct'][-1])}/{samples} latent {record['latent_correct'][-1]} "
            f"{time.time() - t0:.1f}s")
        if (i + 1) % save_every == 0 or i + 1 == len(rows):
            torch.save(record, path)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description="Explicit outputs, thoughts, and latent answers of latent reasoning methods on GSM8K.")
    p.add_argument("--methods", nargs="+", default=METHODS[:4], choices=METHODS)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="exclusive question index (default: end of the split)")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/latent_audit/shards"))
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rows = gsm8k(args.split)
    end = len(rows) if args.end is None else min(args.end, len(rows))
    rows = rows[args.start : end]
    shard = args.out_dir / shard_name(args.split, args.start, end)
    shard.mkdir(parents=True, exist_ok=True)
    logger = Logger("latent_audit_data")
    for name in args.methods:
        path = shard / f"{name}.pt"
        if path.exists() and len(torch.load(path, weights_only=False)["questions"]) == len(rows):
            logger.info(f"{name} {shard.name}: complete, skipped")
            continue
        method = load_method(name, logger, args.device)
        record = run(method, rows, args.split, args.start, args.samples, args.seed, path, logger.info)
        explicit = sum(sum(c) for c in record["explicit_correct"]) / (len(rows) * args.samples)
        latent = [c for c in record["latent_correct"] if c is not None]
        logger.info(f"{name} {shard.name}: explicit accuracy {explicit:.3f}"
                    + (f", latent accuracy {sum(latent) / len(latent):.3f}" if latent else ""))
        del method
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
