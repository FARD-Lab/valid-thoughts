from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

SPLITS = ["train", "val", "test"]
STEP_SLICED = ["soft_thinking", "soft_thinking_noise", "latent_thinking"]
STEPS = [1, 16, 32, 64, 128]
LAMBDAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
DELTAS = [round(0.005 * i, 3) for i in range(81)]
BOOT_RESAMPLES = 200
BOOT_SEED = 0

OUTPUT_REFERENCE = "output_reference"
OUTPUT_EXACT = "output_embedding_exact"
INPUT_RAW = "input_embedding_raw"
BASELINE = "input_embedding"
LIT_LAST = "last_input_token_last_layer"
LIT_MEAN = "last_input_token_layer_mean"
RANDOM_VECTOR = "random_vector"

CACHE_GROUPS: list[tuple[str, list[str]]] = [
    ("last_input_token", [LIT_LAST, LIT_MEAN]),
    *[(tr, [f"{tr}_{n}" for n in STEPS]) for tr in STEP_SLICED],
]
FITTED_NAMES = [name for _, names in CACHE_GROUPS for name in names] + [RANDOM_VECTOR]


class Setup:
    def __init__(self, task_of: torch.Tensor, train, val, test, task_names: list[str]):
        self.task_of = task_of.long()
        self.train, self.val, self.test = (torch.as_tensor(i).long() for i in (train, val, test))
        self.task_names = task_names


def nrm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def comparisons(P: torch.Tensor, C: torch.Tensor, idx: torch.Tensor, task_of: torch.Tensor,
                same: bool) -> dict:
    s = P @ C.T
    own = s.diagonal()
    gap = own[:, None] - s
    tk = task_of[idx]
    mask = (tk[:, None] == tk[None, :]) if same else (tk[:, None] != tk[None, :])
    if same:
        mask &= ~torch.eye(len(idx), dtype=torch.bool)
    m = mask.double()
    wins = []
    for d in DELTAS:
        w = (gap > d).double()
        if d == 0.0:
            w += 0.5 * (gap == 0).double()
        wins.append((w * m).sum(1))
    gi, hi = mask.nonzero(as_tuple=True)
    raw = dict(gap=gap[gi, hi].numpy().astype(np.float32),
               g=idx[gi].numpy().astype(np.int32), h=idx[hi].numpy().astype(np.int32))
    return dict(wins=torch.stack(wins, 1).numpy(), count=m.sum(1).numpy(),
                gap=(gap.double() * m).sum(1).numpy(), own=own.double().numpy(),
                other=(s.double() * m).sum(1).numpy(), raw=raw)


def bootstrap_ids(n: int, B: int = BOOT_RESAMPLES, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def summarize(c: dict, ids: np.ndarray) -> dict:
    count_b = c["count"][ids].sum(1)
    win_b = c["wins"][ids].sum(1) / count_b[:, None]
    gap_b = c["gap"][ids].sum(1) / count_b
    total = c["count"].sum()
    win_se = np.std(win_b, axis=0, ddof=1)
    return dict(
        win=[float(v) for v in c["wins"].sum(0) / total],
        win_se=[float(v) for v in win_se],
        win0_se=float(win_se[0]),
        mean_gap=float(c["gap"].sum() / total), mean_gap_se=float(np.std(gap_b, ddof=1)),
        own_cos=float(c["own"].mean()), other_cos=float(c["other"].sum() / total),
    )


def per_task_scores(c: dict, task_of_test: np.ndarray, task_names: list[str]) -> dict:
    out = {}
    for t, name in enumerate(task_names):
        sel = task_of_test == t
        total = c["count"][sel].sum()
        if total > 0:
            out[name] = float(c["wins"][sel, 0].sum() / total)
    return out


def paired_difference(q: tuple[np.ndarray, np.ndarray], base: tuple[np.ndarray, np.ndarray],
                      ids: np.ndarray) -> dict:
    (w, n), (bw, bn) = q, base
    diff = w.sum() / n.sum() - bw.sum() / bn.sum()
    boot = w[ids].sum(1) / n[ids].sum(1) - bw[ids].sum(1) / bn[ids].sum(1)
    se = float(np.std(boot, ddof=1))
    return dict(diff=float(diff), se=se, z=float(diff / se) if se > 0 else None)


def ridge_predictions(X: torch.Tensor, Y: torch.Tensor, setup: Setup) -> dict:
    Xtr = X[setup.train]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1
    Xs = ((Xtr - mu) / sd).double()
    Ym = Y[setup.train].mean(0).double()
    K = Xs @ Xs.T
    ev, U = torch.linalg.eigh(K)
    UtY = U.T @ (Y[setup.train].double() - Ym)
    Kva, Kte = (((X[i] - mu) / sd).double() @ Xs.T for i in (setup.val, setup.test))
    scale = float(torch.trace(K) / Xs.shape[1])
    out = {}
    for lam in LAMBDAS:
        alpha = U @ (UtY / (ev + lam * scale)[:, None])
        out[lam] = (nrm((Kva @ alpha + Ym).float()), nrm((Kte @ alpha + Ym).float()))
    return out


def same_task_score(P: torch.Tensor, C: torch.Tensor, idx: torch.Tensor, task_of: torch.Tensor) -> float:
    c = comparisons(P, C[idx], idx, task_of, True)
    return float(c["wins"][:, 0].sum() / c["count"].sum())


def score_predictions(setup: Setup, target: torch.Tensor, P_test: torch.Tensor, ids: np.ndarray,
                      name: str, store: dict) -> tuple[dict, dict]:
    C_te = target[setup.test]
    result, qstats = {}, {}
    for setting, same in (("same", True), ("cross", False)):
        c = comparisons(P_test, C_te, setup.test, setup.task_of, same)
        for k, v in c["raw"].items():
            store[f"{name}|{setting}|{k}"] = v
        result[setting] = summarize(c, ids)
        qstats[setting] = (c["wins"][:, 0].copy(), c["count"].copy())
        if same:
            result["per_task_same"] = per_task_scores(c, setup.task_of[setup.test].numpy(),
                                                      setup.task_names)
    store[f"{name}|own"] = (P_test * C_te).sum(1).numpy().astype(np.float32)
    return result, qstats


def evaluate_model(setup: Setup, target: torch.Tensor, references: dict[str, torch.Tensor],
                   baseline: torch.Tensor, fitted: Iterable[tuple[str, torch.Tensor]],
                   log=print) -> tuple[dict, dict]:
    n_test = len(setup.test)
    ids = bootstrap_ids(n_test)
    results: dict[str, dict] = {}
    store: dict[str, np.ndarray] = {"test_ids": setup.test.numpy().astype(np.int32)}

    def fit_and_score(name: str, X: torch.Tensor) -> tuple[dict, dict]:
        preds = ridge_predictions(X, target, setup)
        val = {lam: same_task_score(pv, target, setup.val, setup.task_of) for lam, (pv, _) in preds.items()}
        lam = max(val, key=val.get)
        result, qstats = score_predictions(setup, target, preds[lam][1], ids, name, store)
        result = dict(lam=lam, val_by_lam={str(k): v for k, v in val.items()}, **result)
        log(f"{name} lam {lam} same {result['same']['win'][0]:.4f} cross {result['cross']['win'][0]:.4f}")
        return result, qstats

    base_result, base_q = fit_and_score(BASELINE, baseline)
    for name, P in references.items():
        result, qstats = score_predictions(setup, target, P[setup.test], ids, name, store)
        results[name] = dict(lam=None, **result)
        results[name]["vs_input_embedding"] = {s: paired_difference(qstats[s], base_q[s], ids)
                                               for s in ("same", "cross")}
    results[BASELINE] = base_result
    base_result["vs_input_embedding"] = {s: paired_difference(base_q[s], base_q[s], ids) for s in ("same", "cross")}
    for name, X in fitted:
        result, qstats = fit_and_score(name, X)
        result["vs_input_embedding"] = {s: paired_difference(qstats[s], base_q[s], ids)
                                        for s in ("same", "cross")}
        results[name] = result
    return results, store


def _load_cache(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def reduce_entry(tr_type: str, t: torch.Tensor) -> dict[str, torch.Tensor]:
    t = t.float()
    if tr_type in STEP_SLICED:
        return {f"{tr_type}_{n}": t[:n].mean(0) for n in STEPS}
    if tr_type == "last_input_token":
        return {LIT_LAST: t[-1], LIT_MEAN: t.mean(0)}
    if tr_type == "input_embedding":
        return {BASELINE: t.mean(0)}
    if tr_type == "embedding_no_pooling":
        return {"output_embeddings": t.squeeze(1)}
    raise ValueError(f"unknown thought representation type {tr_type}")


def load_reduced(disc_dir: Path, tr_type: str, n_groups: int,
                 wanted: set[str] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    seen = np.zeros(n_groups, dtype=bool)
    for split in SPLITS:
        cache = _load_cache(disc_dir / f"{split}_other_vectors_{tr_type}.pt")
        for g, t in cache.items():
            g = int(g)
            for name, v in reduce_entry(tr_type, t).items():
                if wanted is not None and name not in wanted:
                    continue
                if name not in out:
                    out[name] = np.empty((n_groups, *v.shape), dtype=np.float32)
                out[name][g] = v.numpy()
            seen[g] = True
        del cache
    if not seen.all():
        raise ValueError(f"{tr_type}: {int((~seen).sum())} of {n_groups} groups are missing")
    return out


def load_split_ids(disc_dir: Path) -> dict[str, list[int]]:
    ids = {}
    for split in SPLITS:
        cache = _load_cache(disc_dir / f"{split}_other_vectors_embedding_no_pooling.pt")
        ids[split] = sorted(int(k) for k in cache)
        del cache
    return ids


def build_task_map(llm_data_dir: Path) -> tuple[list[str], np.ndarray]:
    names, sizes = [], []
    for sub in sorted(d for d in Path(llm_data_dir).iterdir() if d.is_dir() and not d.name.startswith(".")):
        f = sub / "inputs.jsonl"
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            count = sum(1 for _ in fh)
        if count:
            names.append(sub.name)
            sizes.append(count)
    return names, np.repeat(np.arange(len(names)), sizes)


def candidate_stream(disc_dir: Path, n_groups: int, hidden_size: int | None,
                     wanted: set[str] | None, log=print) -> Iterator[tuple[str, torch.Tensor]]:
    for tr_type, names in CACHE_GROUPS:
        names = [x for x in names if wanted is None or x in wanted]
        if not names:
            continue
        if not (disc_dir / f"train_other_vectors_{tr_type}.pt").exists():
            log(f"MISSING {tr_type}, skipped")
            continue
        reduced = load_reduced(disc_dir, tr_type, n_groups, set(names))
        for name in names:
            yield name, torch.from_numpy(reduced.pop(name)).float()
        del reduced
    if wanted is None or RANDOM_VECTOR in wanted:
        vecs = np.stack([np.random.default_rng(g).standard_normal(hidden_size) for g in range(n_groups)])
        yield RANDOM_VECTOR, torch.from_numpy(vecs.astype(np.float32)).float()


def run(disc_dir: Path, llm_data_dir: Path, hidden_size: int | None, out_dir: Path,
        candidates: list[str] | None = None, model: str | None = None, log=print) -> dict:
    t0 = time.time()
    split_ids = load_split_ids(disc_dir)
    all_ids = sorted(sum(split_ids.values(), []))
    n_groups = len(all_ids)
    if all_ids != list(range(n_groups)):
        raise ValueError("group ids over the splits must be exactly 0..N-1")
    task_names, task_of = build_task_map(llm_data_dir)
    if len(task_of) != n_groups:
        raise ValueError(f"llm_data holds {len(task_of)} questions but the caches hold {n_groups} groups")
    setup = Setup(torch.from_numpy(task_of), split_ids["train"], split_ids["val"], split_ids["test"], task_names)

    outputs = load_reduced(disc_dir, "embedding_no_pooling", n_groups)["output_embeddings"]
    outputs = torch.from_numpy(outputs).float()
    target = nrm(nrm(outputs).mean(1))
    exact = nrm(outputs[:, 0])
    del outputs
    ie = torch.from_numpy(load_reduced(disc_dir, "input_embedding", n_groups)[BASELINE]).float()
    if hidden_size is None:
        hidden_size = ie.shape[1]
    references = {OUTPUT_REFERENCE: target, OUTPUT_EXACT: exact, INPUT_RAW: nrm(ie)}
    wanted = None if candidates is None else set(candidates)
    results, store = evaluate_model(setup, target, references, ie,
                                    candidate_stream(disc_dir, n_groups, hidden_size, wanted, log), log)
    out = dict(model=model or disc_dir.name, deltas=DELTAS, lambdas=LAMBDAS,
               bootstrap=dict(resamples=BOOT_RESAMPLES, seed=BOOT_SEED),
               n_questions={s: len(v) for s, v in split_ids.items()}, tasks=task_names,
               candidates=results)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "separability.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(out_dir / "separability_gaps.npz", **store)
    log(f"done {time.time() - t0:.0f}s")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Linear separability of thought representations against question-average outputs.")
    p.add_argument("--disc-data-dir", type=Path, required=True,
                   help="cache directory with {split}_other_vectors_*.pt")
    p.add_argument("--llm-data-dir", type=Path, required=True,
                   help="Phase 1 output with one subdirectory per task")
    p.add_argument("--hidden-size", type=int, default=None,
                   help="dimension of the random vector (default: width of the input embedding)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--candidates", nargs="+", default=None, choices=FITTED_NAMES,
                   help="restrict the fitted candidates (references and input_embedding are always scored)")
    p.add_argument("--model", default=None, help="label stored in the JSON")
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    run(args.disc_data_dir, args.llm_data_dir, args.hidden_size, args.out_dir, args.candidates, args.model)


if __name__ == "__main__":
    main()
