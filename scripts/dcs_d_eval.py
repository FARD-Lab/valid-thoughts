from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MODEL_CONFIGS: dict[str, dict] = {
    "llama_8b":        {"disc_data_dir": "outputs/disc_data_llama_8b",          "rv_dim": 4096},
    "llama_70b":       {"disc_data_dir": "outputs/disc_data_llama_70b",         "rv_dim": 8192},
    "deepseek_r1_32b": {"disc_data_dir": "outputs/disc_data_deepseek_r1_32b",   "rv_dim": 5120},
    "skywork_or1_32b": {"disc_data_dir": "outputs/disc_data_skywork_or1_32b",   "rv_dim": 5120},
    "gpt_oss_20b":     {"disc_data_dir": "outputs/disc_data_gpt_oss_20b",       "rv_dim": 2880},
}

SPLITS      = ["train", "val", "test"]
STEP_SLICED = {"soft_thinking", "soft_thinking_noise", "latent_thinking"}
PER_BEAM    = {"embedding_no_pooling"}
STEPS       = [1, 16, 32, 64, 128]
TAU_VALUES  = [0.7, 0.8, 0.85, 0.9, 0.95]

TR_CELLS: list[tuple[str, list[int]]] = [
    ("last_input_token",        [1]),
    ("last_input_hidden_state", [1]),
    ("soft_thinking",           STEPS),
    ("soft_thinking_noise",     STEPS),
    ("latent_thinking",         STEPS),
    ("embedding_no_pooling",    [1]),
    ("embedding_pooling",       [1]),
    ("input_embedding",         [1]),
    ("random_vector",           [1]),
]

def _semantic_entropy(sim: torch.Tensor, tau: float) -> float:
    K = sim.shape[0]
    parent = list(range(K))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i in range(K):
        for j in range(i + 1, K):
            if sim[i, j].item() > tau:
                union(i, j)

    sizes: dict[int, int] = {}
    for i in range(K):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1

    p = np.array(list(sizes.values()), dtype=float) / K
    return float(-np.sum(p * np.log(p + 1e-10)))

def load_embedding_sims(disc_data_dir: Path) -> dict[int, torch.Tensor]:
    import torch.nn.functional as F
    sims: dict[int, torch.Tensor] = {}
    for split in SPLITS:
        cf = disc_data_dir / f"{split}_other_vectors_embedding_no_pooling.pt"
        if not cf.exists():
            continue
        cache = torch.load(cf, map_location="cpu", weights_only=False)
        for g_idx, t in cache.items():
            emb = F.normalize(t.squeeze(1).float(), dim=-1)
            sims[int(g_idx)] = emb @ emb.T
    return sims

def compute_H_map(sims: dict[int, torch.Tensor], tau: float) -> dict[int, float]:
    return {g: _semantic_entropy(s, tau) for g, s in sims.items()}

def load_tr_vectors(
    disc_data_dir: Path,
    tr_type: str,
    think_steps: int,
    g_indices: list[int],
    rv_dim: int,
) -> tuple[np.ndarray | None, list[int] | None]:
    if tr_type == "last_input_hidden_state":
        merged: dict[int, np.ndarray] = {}
        for split in SPLITS:
            cf = disc_data_dir / f"{split}_other_vectors_last_input_token.pt"
            if not cf.exists():
                continue
            for g_idx, t in torch.load(cf, map_location="cpu", weights_only=False).items():
                merged[int(g_idx)] = t.float()[-1].numpy()
        pairs = [(merged[g], g) for g in g_indices if g in merged]
        if not pairs:
            return None, None
        vecs, valid = zip(*pairs)
        return np.array(vecs, dtype=np.float32), list(valid)

    if tr_type == "random_vector":
        vecs = [np.random.default_rng(int(g)).standard_normal(rv_dim).astype(np.float32)
                for g in g_indices]
        return np.array(vecs, dtype=np.float32), list(g_indices)

    merged_t: dict[int, torch.Tensor] = {}
    for split in SPLITS:
        cf = disc_data_dir / f"{split}_other_vectors_{tr_type}.pt"
        if not cf.exists():
            continue
        for g_idx, t in torch.load(cf, map_location="cpu", weights_only=False).items():
            merged_t[int(g_idx)] = t.float()

    if not merged_t:
        return None, None

    vecs, valid = [], []
    for g_idx in g_indices:
        if g_idx not in merged_t:
            continue
        t = merged_t[g_idx]
        if tr_type in STEP_SLICED:
            t = t[:think_steps].mean(dim=0)
        elif tr_type in PER_BEAM:
            t = t.squeeze(1).mean(dim=0)
        else:
            t = t.mean(dim=0)
        vecs.append(t.numpy())
        valid.append(g_idx)

    return (np.array(vecs, dtype=np.float32), valid) if vecs else (None, None)

def dom_auroc_cv(
    T: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs: list[float] = []

    for train_idx, test_idx in skf.split(T, labels):
        T_tr, T_te = T[train_idx], T[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        if y_tr.sum() == 0 or (1 - y_tr).sum() == 0:
            aurocs.append(0.5)
            continue

        mu_pos = T_tr[y_tr == 1].mean(axis=0)
        mu_neg = T_tr[y_tr == 0].mean(axis=0)
        direction = mu_pos - mu_neg
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            aurocs.append(0.5)
            continue

        scores = T_te @ (direction / norm)

        if y_te.sum() == 0 or (1 - y_te).sum() == 0:
            aurocs.append(0.5)
            continue

        aurocs.append(float(roc_auc_score(y_te, scores)))

    return float(np.mean(aurocs)), float(np.std(aurocs))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    type=Path, default=Path("outputs/dcs_d_results.csv"))
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "tr_type", "think_steps", "tau",
        "auroc", "auroc_std", "n", "n_h_nonzero",
    ]
    rows: list[dict] = []

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        disc_data_dir = Path(cfg["disc_data_dir"])
        rv_dim: int = cfg["rv_dim"]

        if not disc_data_dir.exists():
            print(f"[{model_name}] disc_data_dir missing, skipping")
            continue

        print(f"\n[{model_name}] loading embedding similarities …")
        sims = load_embedding_sims(disc_data_dir)
        if not sims:
            print(f"[{model_name}] no embedding_no_pooling cache found, skipping")
            continue
        g_indices = sorted(sims.keys())
        print(f"[{model_name}]  n={len(g_indices)} questions")

        print(f"[{model_name}] loading TR vectors …")
        tr_cache: dict[tuple[str, int], tuple[np.ndarray, list[int]]] = {}
        for tr_type, steps in TR_CELLS:
            for step in steps:
                T, valid_g = load_tr_vectors(disc_data_dir, tr_type, step, g_indices, rv_dim)
                if T is not None:
                    tr_cache[(tr_type, step)] = (T, valid_g)
                    print(f"  cached  {tr_type:35s} steps={step:3d}  n={len(valid_g)}")

        for tau in TAU_VALUES:
            H_map = compute_H_map(sims, tau)
            n_nonzero = sum(1 for v in H_map.values() if v > 1e-6)
            print(f"\n  τ={tau:.2f}  H>0={n_nonzero}/{len(g_indices)}")

            for tr_type, steps in TR_CELLS:
                for step in steps:
                    key = (tr_type, step)
                    if key not in tr_cache:
                        continue
                    T, valid_g = tr_cache[key]
                    H = np.array([H_map[g] for g in valid_g])
                    labels = (H > 1e-6).astype(int)
                    n_pos = int(labels.sum())

                    if n_pos < 10 or (len(labels) - n_pos) < 10:
                        continue

                    auroc_mean, auroc_std = dom_auroc_cv(T, labels)
                    rows.append({
                        "model": model_name, "tr_type": tr_type,
                        "think_steps": step, "tau": tau,
                        "auroc": round(auroc_mean, 6),
                        "auroc_std": round(auroc_std, 6),
                        "n": len(valid_g), "n_h_nonzero": n_pos,
                    })
                    print(
                        f"    {tr_type:35s} steps={step:3d}  "
                        f"AUROC={auroc_mean:.4f}±{auroc_std:.4f}  "
                        f"H>0={n_pos}"
                    )

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {args.out}")

if __name__ == "__main__":
    main()
