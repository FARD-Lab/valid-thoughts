import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.causality_native_eval import written_pass
from scripts.dcs_d_eval import TAU_VALUES, _semantic_entropy, dom_auroc_cv
from scripts.minimality_native_eval import chat_segments, plain_segments, score_blocks
from scripts.native_common import (
    bootstrap_mean,
    ce_and_kl,
    chunk_blocks,
    embed_ids,
    embedding_norm,
    information_ratios,
    packed_hidden,
    random_vectors,
    recovered_fractions,
)
from scripts.separability_eval import (
    DELTAS,
    Setup,
    bootstrap_ids,
    comparisons,
    nrm,
    paired_difference,
    ridge_predictions,
    same_task_score,
)
from src.model.latent_methods import load_method
from src.utils.config import ModelType
from src.utils.logging import Logger

MEASURES = ["separability", "stability", "causality", "minimality"]
STEPS = [1, 16, 32, 64, 128]
WINDOWS = [16, 32]
OUTPUT = "output_embedding"
INPUT = "input_embedding"
RANDOM = "random_vector"


def load(data_dir: Path, name: str) -> dict:
    return torch.load(data_dir / f"{name}.pt", weights_only=False)


def cells(record: dict) -> dict[str, list[torch.Tensor]]:
    out = {}
    for family, vecs in record["thoughts"].items():
        if family in ("soft_thinking", "soft_thinking_noise", "latent_thinking"):
            for s in STEPS:
                out[f"{family}_{s}"] = [v[:s] for v in vecs]
        else:
            out[family] = vecs
    return out


def pooled(record: dict) -> dict[str, torch.Tensor]:
    if "pooled" in record:
        return record["pooled"]
    return {name: torch.stack([v.mean(0) for v in vecs]) for name, vecs in cells(record).items()}


def random_rows(n: int, dim: int) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.random.default_rng(g).standard_normal(dim) for g in range(n)]).astype(np.float32))


def cv_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def fold_wins(X, C, folds, fitted):
    n = C.shape[0]
    task_of = torch.zeros(n, dtype=torch.long)
    wins, count, lams = np.zeros((n, len(DELTAS))), np.zeros(n), []
    for k, test in enumerate(folds):
        v = (k + 1) % len(folds)
        train = np.sort(np.concatenate([f for j, f in enumerate(folds) if j not in (k, v)]))
        setup = Setup(task_of, train, folds[v], test, ["gsm8k"])
        if fitted:
            preds = ridge_predictions(X, C, setup)
            val = {lam: same_task_score(pv, C, setup.val, task_of) for lam, (pv, _) in preds.items()}
            lam = max(val, key=val.get)
            lams.append(lam)
            P = preds[lam][1]
        else:
            P = X[setup.test]
        c = comparisons(P, C[setup.test], setup.test, task_of, True)
        wins[test], count[test] = c["wins"], c["count"]
    return wins, count, lams


def separability(record: dict, emb: dict, folds: list[np.ndarray]) -> dict:
    outputs = emb[record["method"]]
    C = nrm(nrm(outputs).mean(1))
    ie = emb["questions"]
    thoughts = pooled(record)
    dim = next(iter(thoughts.values())).shape[1]
    ids = bootstrap_ids(C.shape[0])
    base = fold_wins(ie, C, folds, True)
    rows = {"output_reference": (C, False), "output_embedding_exact": (nrm(outputs[:, 0]), False),
            "input_embedding_raw": (ie, False), INPUT: (ie, True), RANDOM: (random_rows(C.shape[0], dim), True)}
    rows |= {name: (X, True) for name, X in thoughts.items()}
    out = {}
    for name, (X, fitted) in rows.items():
        wins, count, lams = base if name == INPUT else fold_wins(X, C, folds, fitted)
        curve = wins.sum(0) / count.sum()
        boot = wins[ids, 0].sum(1) / count[ids].sum(1)
        out[name] = dict(score=float(curve[0]), se=float(np.std(boot, ddof=1)), curve=[float(x) for x in curve],
                         lams=lams, vs_input_embedding=paired_difference((wins[:, 0], count), (base[0][:, 0], base[1]), ids))
    return out


def stability(record: dict, emb: dict, seed: int) -> dict:
    outputs = nrm(emb[record["method"]])
    sims = outputs @ outputs.transpose(1, 2)
    thoughts = pooled(record)
    dim = next(iter(thoughts.values())).shape[1]
    rows = {OUTPUT: outputs.mean(1), INPUT: emb["questions"], RANDOM: random_rows(sims.shape[0], dim)} | thoughts
    out = {}
    for tau in TAU_VALUES:
        labels = np.array([_semantic_entropy(s, tau) > 1e-6 for s in sims], dtype=int)
        cell = dict(n_diverged=int(labels.sum()), n=len(labels), auroc={})
        if min(labels.sum(), len(labels) - labels.sum()) >= 10:
            for name, X in rows.items():
                mean, std = dom_auroc_cv(X.numpy(), labels, seed=seed)
                cell["auroc"][name] = dict(mean=mean, std=std)
        out[str(tau)] = cell
    return out


def norm_matched(t: torch.Tensor, seed: int) -> torch.Tensor:
    vecs = torch.randn(t.shape, generator=torch.Generator().manual_seed(seed))
    return vecs * (t.norm(dim=1, keepdim=True) / vecs.norm(dim=1, keepdim=True).clamp_min(1e-12))


def qid(record: dict, g: int) -> int:
    return record["index"][g] if "index" in record else g


def arms(method, thoughts: dict, g: int, partner: int, norm: float, seed: int) -> list[tuple[str, torch.Tensor]]:
    out = []
    trained = method.name != "llama"
    for name, vecs in thoughts.items():
        t = vecs[g]
        out.append((name, t))
        if trained or name.endswith(f"_{STEPS[-1]}"):
            out.append((f"swap_{name}", vecs[partner]))
        if trained:
            out.append((f"random_{name}", norm_matched(t, seed)))
    if not trained:
        for s in STEPS:
            out.append((f"random_{s}", random_vectors(s, method.model.config.hidden_size, norm, seed)))
    return out


def targets(method, ids: list[int]) -> list[tuple[str, int]]:
    out = []
    start = method.answer_start(ids)
    if start is not None and start < len(ids):
        out.append(("answer", start))
    if method.name == "llama":
        out += [(f"last_{n}", len(ids) - n) for n in WINDOWS if len(ids) >= n + 1]
    return out


def half(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


def causality_question(method, record, thoughts, g, partner, norm, seed, cap):
    model = method.model
    question, outputs = record["questions"][g], record["output_ids"][g]
    items = [(b, label, start) for b, ids in enumerate(outputs) for label, start in targets(method, ids)]
    if not items:
        return []
    live = sorted({b for b, _, _ in items})
    tail = max(len(outputs[b]) - start for b, _, start in items)
    explicit_prefix = embed_ids(model, method.explicit_prompt(question))
    ce_all, tails = written_pass(model, explicit_prefix, [outputs[b] for b in live], tail, cap)
    ce_of, tail_of = dict(zip(live, ce_all)), dict(zip(live, tails))
    recs = []
    for b, label, start in items:
        n = len(outputs[b]) - start
        recs.append(dict(g_idx=g, beam_idx=b, window=label, n_tokens=n, explicit_ce=half(ce_of[b][-n:]), arms={}))
    post = method.post_ids
    blocks = [post + outputs[b][start:] for b, _, start in items]
    refs = [tail_of[b][-(len(outputs[b]) - start):] for b, _, start in items]
    prefix = embed_ids(model, method.latent_prompt(question))
    for name, lead in [("question", None)] + arms(method, thoughts, g, partner, norm, seed + qid(record, g)):
        lead = None if lead is None else lead[None].float()
        width = 0 if lead is None else lead.size(1)
        for chunk in chunk_blocks([len(x) for x in blocks], prefix.size(0) + width, cap):
            hidden = packed_hidden(model, prefix, lead, [blocks[i] for i in chunk])
            for i, h in zip(chunk, hidden):
                ce, kl = ce_and_kl(model, h[:, len(post):], torch.as_tensor(blocks[i][len(post):]), refs[i])
                recs[i]["arms"][name] = dict(ce=half(ce[0]), kl=half(kl[0]))
    return recs


def summarize_causality(records, n_questions, resamples, seed) -> dict:
    out = {}
    for label in sorted({r["window"] for r in records}):
        sums, counts = {}, np.zeros(n_questions)
        for r in records:
            if r["window"] != label:
                continue
            counts[r["g_idx"]] += 1
            entries = {"written": {"ce": r["explicit_ce"]}} | r["arms"]
            for arm, vals in entries.items():
                for key, arr in vals.items():
                    sums.setdefault(f"{arm}|{key}", np.zeros(n_questions))[r["g_idx"]] += float(np.mean(arr, dtype=np.float64))
        cell = {}
        for name, s in sums.items():
            arm, key = name.split("|")
            mean, se = bootstrap_mean(s, counts, resamples, seed)
            cell.setdefault(arm, {})[key] = mean
            cell[arm][f"{key}_se"] = se
        for arm, vals in recovered_fractions(sums, counts, resamples, seed).items():
            cell[arm].update(vals)
        out[label] = dict(n_outputs=int(counts.sum()), n_questions=int((counts > 0).sum()), arms=cell)
    return out


def minimality_question(method, record, thoughts, g, norm, seed, cap, segments):
    model, tokenizer = method.model, method.tokenizer
    before, after, between = segments
    pre = tokenizer(before).input_ids + method.open_ids
    post = method.close_ids + tokenizer(after, add_special_tokens=False).input_ids
    mid = tokenizer(between, add_special_tokens=False).input_ids
    x = tokenizer(record["questions"][g], add_special_tokens=False).input_ids
    ys = [tokenizer(t, add_special_tokens=False).input_ids for t in record["outputs"][g]]
    beams = [b for b, y in enumerate(ys) if y]
    recs = {b: dict(g_idx=g, beam_idx=b, n_y=len(ys[b]), n_x=len(x), arms={}) for b in beams}
    blocks = {b: post + ys[b] + mid + x for b in beams}
    spans = {}
    for b in beams:
        y0 = len(post)
        x0 = y0 + len(ys[b]) + len(mid)
        spans[b] = ((y0, y0 + len(ys[b])), (x0, x0 + len(x)))
    prefix = embed_ids(model, pre)

    def run(name, lead, members):
        width = 0 if lead is None else lead.size(1)
        for chunk in chunk_blocks([len(blocks[b]) for b in members], len(pre) + width, cap):
            chosen = [members[i] for i in chunk]
            scored = score_blocks(model, prefix, lead, [blocks[b] for b in chosen], [spans[b] for b in chosen])
            for b, (ce_y, ce_x) in zip(chosen, scored):
                recs[b]["arms"][name] = dict(ce_y=half(ce_y[0]), ce_x=half(ce_x[0]))

    run("question", embed_ids(model, x)[None], beams)
    run("empty", None, beams)
    for b in beams:
        run("output", embed_ids(model, ys[b])[None], [b])
    for name, lead in arms(method, thoughts, g, g, norm, seed + qid(record, g)):
        if not name.startswith("swap_"):
            run(name, lead[None].float(), beams)
    return [recs[b] for b in beams]


def summarize_minimality(records, n_questions, resamples, seed) -> dict:
    sums, counts = {}, np.zeros(n_questions)
    for r in records:
        counts[r["g_idx"]] += 1
        for arm, vals in r["arms"].items():
            ce_y = float(np.mean(vals["ce_y"], dtype=np.float64))
            ce_x = float(np.mean(vals["ce_x"], dtype=np.float64))
            for key, value in (("ce_y", ce_y), ("ce_x", ce_x), ("delta_ib", ce_x - ce_y)):
                sums.setdefault(f"{arm}|{key}", np.zeros(n_questions))[r["g_idx"]] += value
    out = {}
    for name, s in sums.items():
        arm, key = name.split("|")
        mean, se = bootstrap_mean(s, counts, resamples, seed)
        out.setdefault(arm, {})[key] = mean
        out[arm][f"{key}_se"] = se
    empty, question, output = out["empty"], out["question"], out["output"]
    for arm, vals in information_ratios(sums, counts, resamples, seed).items():
        out[arm].update(vals)
    out["ranges"] = dict(y=empty["ce_y"] - output["ce_y"], x=empty["ce_x"] - question["ce_x"])
    return out


def native(measure: str, name: str, record: dict, args, logger) -> dict:
    path = args.data_dir / f"{measure}_{name}.pt"
    partial = path.with_name(f"{path.stem}.partial.pt")
    n = len(record["questions"])
    if args.reuse_records and path.exists():
        records = torch.load(path, weights_only=False)
    else:
        done, records = 0, []
        if partial.exists():
            state = torch.load(partial, weights_only=False)
            done, records = state["done"], state["records"]
        method = load_method(name, logger, args.device)
        thoughts = cells(record)
        norm = embedding_norm(method.model)
        segments = (chat_segments(types.SimpleNamespace(tokenizer=method.tokenizer, model_type=ModelType.LLAMA))
                    if method.chat_template else plain_segments())
        with torch.no_grad():
            for g in tqdm(range(done, n), desc=f"{measure} {name}", unit="question"):
                if measure == "causality":
                    records += causality_question(method, record, thoughts, g, (g + 1) % n, norm, args.seed,
                                                  args.max_row_tokens)
                else:
                    records += minimality_question(method, record, thoughts, g, norm, args.seed, args.max_row_tokens,
                                                   segments)
                if (g + 1) % args.save_every == 0 and g + 1 < n:
                    torch.save(dict(done=g + 1, records=records), partial)
        torch.save(records, path)
        partial.unlink(missing_ok=True)
    summarize = summarize_causality if measure == "causality" else summarize_minimality
    return summarize(records, n, args.bootstrap_resamples, args.seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Separability, stability, causality, and minimality of latent reasoning methods on GSM8K.")
    p.add_argument("--data-dir", type=Path, default=Path("outputs/latent_audit"))
    p.add_argument("--methods", nargs="+", default=["codi", "sim_cot", "colar", "llama"])
    p.add_argument("--measures", nargs="+", default=MEASURES, choices=MEASURES)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-row-tokens", type=int, default=4096)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reuse-records", action="store_true",
                   help="summarize saved causality and minimality records instead of recomputing them")
    p.add_argument("--save-every", type=int, default=25,
                   help="questions between checkpoints of the causality and minimality records, which a rerun resumes from")
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    logger = Logger("latent_audit_eval")
    emb = torch.load(args.data_dir / "embeddings.pt", weights_only=False) if {"separability", "stability"} & set(args.measures) else None
    for measure in args.measures:
        path = args.data_dir / f"{measure}.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        for name in args.methods:
            record = load(args.data_dir, name)
            if measure == "separability":
                results[name] = separability(record, emb, cv_folds(len(record["questions"]), args.folds, args.seed))
            elif measure == "stability":
                results[name] = stability(record, emb, args.seed)
            else:
                results[name] = native(measure, name, record, args, logger)
            path.write_text(json.dumps(results, indent=1))
            logger.info(f"{measure} {name} written")


if __name__ == "__main__":
    main()
