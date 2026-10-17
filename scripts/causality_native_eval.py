import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from scripts.native_common import (
    bootstrap_mean,
    ce_and_kl,
    chunk_blocks,
    embed_ids,
    embedding_norm,
    encode_first,
    encode_rest,
    explicit_ce_and_tail,
    load_model,
    load_outputs,
    load_questions,
    load_thoughts,
    packed_hidden,
    prompt_text,
    random_vectors,
    recovered_fractions,
)
from scripts.separability_eval import build_task_map
from src.utils import set_seed_all
from src.utils.config import CausalityNativeEvalConfig, register_configs
from src.utils.logging import Logger

register_configs()


def make_pairs(groups: list[int], task_of: np.ndarray, seed: int) -> tuple[dict[int, int], dict[int, int]]:
    rng = np.random.default_rng(seed)
    groups_arr = np.asarray(sorted(groups))
    tasks = task_of[groups_arr]
    same: dict[int, int] = {}
    cross: dict[int, int] = {}
    for t in np.unique(tasks):
        members = rng.permutation(groups_arr[tasks == t])
        if len(members) < 2:
            raise ValueError(f"task {t} has {len(members)} test question, a same-task partner needs two")
        for i, g in enumerate(members):
            same[int(g)] = int(members[(i + 1) % len(members)])
    for g, t in zip(groups_arr, tasks):
        others = groups_arr[tasks != t]
        if len(others) == 0:
            raise ValueError("a cross-task partner needs test questions from at least two tasks")
        cross[int(g)] = int(rng.choice(others))
    return same, cross


def arm_leads(thoughts, families, g, partner, rv, steps, swap, random):
    names, vecs = [], []
    for family in families:
        names.append(f"{family}_{steps}")
        vecs.append(thoughts[family][g][:steps])
    if swap:
        for family in families:
            names.append(f"swap_{family}_{steps}")
            vecs.append(thoughts[family][partner][:steps])
    if random:
        names.append(f"random_{steps}")
        vecs.append(rv[:steps])
    return names, torch.stack([v.float() for v in vecs])


def written_pass(model, prefix, outputs, tail, cap):
    ce, tails = [None] * len(outputs), [None] * len(outputs)
    for chunk in chunk_blocks([len(y) for y in outputs], prefix.size(0), cap):
        hidden = packed_hidden(model, prefix, None, [outputs[i] for i in chunk])
        for i, h in zip(chunk, hidden):
            ce[i], tails[i] = explicit_ce_and_tail(model, h, torch.as_tensor(outputs[i]), min(tail, len(outputs[i])))
    return ce, tails


def window_pass(model, prefix, lead, blocks, refs):
    hidden = packed_hidden(model, prefix, lead, blocks)
    return [ce_and_kl(model, h, torch.as_tensor(block), ref) for h, block, ref in zip(hidden, blocks, refs)]


def to_half(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


def summarize(records, groups, windows, resamples, seed):
    index = {g: i for i, g in enumerate(groups)}
    out = {}
    for n in windows:
        sums: dict[str, np.ndarray] = {}
        counts = np.zeros(len(groups))
        for r in records:
            w = r["windows"].get(n)
            if w is None:
                continue
            i = index[r["g_idx"]]
            counts[i] += 1
            entries = {"written": {"ce": r["explicit_ce"][-n:]}} | w
            for arm, vals in entries.items():
                for key, arr in vals.items():
                    name = f"{arm}|{key}"
                    sums.setdefault(name, np.zeros(len(groups)))[i] += float(np.mean(arr, dtype=np.float64))
        cell = {}
        for name, s in sums.items():
            arm, key = name.split("|")
            mean, se = bootstrap_mean(s, counts, resamples, seed)
            cell.setdefault(arm, {})[key] = mean
            cell[arm][f"{key}_se"] = se
        if counts.any():
            for arm, vals in recovered_fractions(sums, counts, resamples, seed).items():
                cell[arm].update(vals)
        out[str(n)] = dict(n_outputs=int(counts.sum()), n_questions=int((counts > 0).sum()), arms=cell)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="causality_native_eval")
def main(cfg: CausalityNativeEvalConfig) -> None:
    logger = Logger.from_config(cfg.logging)
    logger.info("Causality Native Eval")
    logger.info(OmegaConf.to_yaml(cfg))
    set_seed_all(cfg.seed)

    llm = load_model(cfg.base_llm, logger)
    model, tokenizer = llm.model, llm.tokenizer
    hidden_size = model.config.hidden_size
    norm = embedding_norm(model)

    families = list(cfg.families)
    steps = sorted(int(s) for s in cfg.steps)
    windows = sorted(int(w) for w in cfg.windows)
    questions = load_questions(cfg.llm_data_dir)
    _, task_of = build_task_map(Path(cfg.llm_data_dir))
    outputs = load_outputs(cfg.tr_data_dir, llm.model_type)
    groups = sorted(outputs)
    if groups[-1] >= len(questions):
        raise ValueError(f"llm_data holds {len(questions)} questions but the test split holds group {groups[-1]}")
    same, _ = make_pairs(groups, task_of, cfg.seed)
    thoughts = load_thoughts(cfg.tr_data_dir, families, hidden_size, logger)
    eval_groups = groups[: cfg.num_examples] if cfg.num_examples > 0 else groups

    records = []
    with torch.no_grad():
        for g in tqdm(eval_groups, desc="causality native", unit="question"):
            prefix = embed_ids(model, encode_first(tokenizer, prompt_text(llm, questions[g])))
            rv = random_vectors(steps[-1], hidden_size, norm, cfg.seed + g)
            texts = outputs[g][: cfg.beams] if cfg.beams > 0 else outputs[g]
            ys = [encode_rest(tokenizer, t) for t in texts]
            live = [b for b, y in enumerate(ys) if len(y) >= 2]
            recs = {b: dict(g_idx=g, beam_idx=b, n_tokens=len(y), windows={}, explicit_ce=np.zeros(0, dtype=np.float16))
                    for b, y in enumerate(ys)}
            ce_all, tails = written_pass(model, prefix, [ys[b] for b in live], windows[-1], cfg.max_row_tokens)
            ref_tail = {}
            for b, ce, tail in zip(live, ce_all, tails):
                recs[b]["explicit_ce"] = to_half(ce)
                ref_tail[b] = tail
            items = [(b, n) for b in live for n in windows if len(ys[b]) >= n + 1]
            blocks = [ys[b][-n:] for b, n in items]
            refs = [ref_tail[b][-n:] for b, n in items]
            for chunk in chunk_blocks([len(x) for x in blocks], prefix.size(0) + steps[-1], cfg.max_row_tokens):
                chunk_blocks_list = [blocks[i] for i in chunk]
                chunk_refs = [refs[i] for i in chunk]
                results = {"question": [(ce[0], kl[0]) for ce, kl in
                                        window_pass(model, prefix, None, chunk_blocks_list, chunk_refs)]}
                for s in steps:
                    names, leads = arm_leads(thoughts, families, g, same[g], rv, s, cfg.swap, cfg.random_vector)
                    per_block = window_pass(model, prefix, leads, chunk_blocks_list, chunk_refs)
                    for a, name in enumerate(names):
                        results[name] = [(ce[a], kl[a]) for ce, kl in per_block]
                for pos, i in enumerate(chunk):
                    b, n = items[i]
                    arms = recs[b]["windows"].setdefault(n, {})
                    for name, per_block in results.items():
                        ce, kl = per_block[pos]
                        arms[name] = dict(ce=to_half(ce), kl=to_half(kl))
            records.extend(recs[b] for b in sorted(recs))

    summary = summarize(records, eval_groups, windows, cfg.bootstrap_resamples, cfg.seed)
    for n, s in summary.items():
        logger.info(f"window {n}: {s['n_outputs']} outputs")
        for arm, vals in s["arms"].items():
            kl = vals.get("kl")
            logger.info(f"  {arm:32s} ce {vals['ce']:.3f}" + ("" if kl is None else f"  kl {kl:.3f}"))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(model=cfg.model_label or cfg.base_llm.name, families=families, steps=steps, windows=windows,
                beams=cfg.beams, swap=cfg.swap, seed=cfg.seed,
                bootstrap=dict(resamples=cfg.bootstrap_resamples, seed=cfg.seed),
                same_task_partner={str(g): same[g] for g in eval_groups})
    (out_dir / "causality_native_summary.json").write_text(json.dumps(meta | dict(windows_summary=summary), indent=1))
    torch.save(dict(meta=meta, records=records), out_dir / "causality_native.pt")
    logger.info(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
