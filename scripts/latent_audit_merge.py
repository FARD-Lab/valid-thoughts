import argparse
from pathlib import Path

import torch

from scripts.latent_audit_eval import pooled

LISTS = ["index", "questions", "gold", "output_ids", "outputs", "explicit_correct", "latent_answers", "latent_correct"]
NATIVE = ["causality", "minimality"]


def shard_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir())


def merge_method(shards: list[Path], name: str) -> tuple[dict, list[int]]:
    merged, sizes = None, []
    for shard in shards:
        record = torch.load(shard / f"{name}.pt", weights_only=False)
        if merged is None:
            merged = {k: v for k, v in record.items() if k not in LISTS + ["thoughts", "split"]}
            merged |= {k: [] for k in LISTS} | dict(splits=[], pooled={})
        for key in LISTS:
            merged[key] += record[key]
        sizes.append(len(record["questions"]))
        merged["splits"] += [record["split"]] * sizes[-1]
        for cell, vecs in pooled(record).items():
            merged["pooled"].setdefault(cell, []).append(vecs)
    merged["pooled"] = {cell: torch.cat(parts) for cell, parts in merged["pooled"].items()}
    return merged, sizes


def merge_records(shards: list[Path], measure: str, name: str, sizes: list[int], log):
    missing = [s.name for s in shards if not (s / f"{measure}_{name}.pt").exists()]
    if missing:
        log(f"{measure} {name}: skipped, {len(missing)} shards have no records ({', '.join(missing[:5])})")
        return None
    out, offset = [], 0
    for shard, size in zip(shards, sizes):
        for r in torch.load(shard / f"{measure}_{name}.pt", weights_only=False):
            out.append(r | dict(g_idx=r["g_idx"] + offset))
        offset += size
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge latent audit shards into one directory for the summary evaluation.")
    p.add_argument("--shards-dir", type=Path, default=Path("outputs/latent_audit/shards"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/latent_audit/merged"))
    p.add_argument("--methods", nargs="+", default=["codi", "sim_cot", "colar", "llama"])
    args = p.parse_args()
    log = print
    shards = shard_dirs(args.shards_dir)
    incomplete = [s.name for s in shards
                  if not all((s / f"{m}.pt").exists() for m in args.methods) or not (s / "embeddings.pt").exists()]
    if incomplete:
        raise SystemExit(f"shards without data or embeddings: {', '.join(incomplete)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.methods:
        merged, sizes = merge_method(shards, name)
        torch.save(merged, args.out_dir / f"{name}.pt")
        log(f"{name}: {len(merged['questions'])} questions from {len(shards)} shards")
        for measure in NATIVE:
            records = merge_records(shards, measure, name, sizes, log)
            if records is not None:
                torch.save(records, args.out_dir / f"{measure}_{name}.pt")
    embeddings = [torch.load(s / "embeddings.pt", weights_only=False) for s in shards]
    out = dict(model=embeddings[0]["model"], questions=torch.cat([e["questions"] for e in embeddings]))
    for name in args.methods:
        out[name] = torch.cat([e[name] for e in embeddings])
    torch.save(out, args.out_dir / "embeddings.pt")


if __name__ == "__main__":
    main()
