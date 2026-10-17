import json
from pathlib import Path

import numpy as np
import torch
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward as gpt_oss_eager

from src.dataset.bbeh_loader import BBEHLoader
from src.dataset.tr_loader import ThoughtRepresentationLoader
from src.model.llm import BaseLLM
from src.utils.config import ModelType, ThoughtRepresentation

def load_questions(llm_data_dir) -> list[str]:
    texts: list[str] = []
    for sub in sorted(d for d in Path(llm_data_dir).iterdir() if d.is_dir() and not d.name.startswith(".")):
        path = sub / "inputs.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        rows.sort(key=lambda r: int(r["group_idx"]))
        if [int(r["group_idx"]) for r in rows] != list(range(len(rows))):
            raise ValueError(f"{path}: group_idx is not 0..{len(rows) - 1}")
        texts.extend(r["input_text"] for r in rows)
    return texts


HARMONY_ANALYSIS = "<|channel|>analysis<|message|>"
HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_TURNS = {"assistantfinal": "<|end|><|start|>assistant" + HARMONY_FINAL,
                 "assistantanalysis": "<|end|><|start|>assistant" + HARMONY_ANALYSIS}


def harmony_output(text: str) -> str:
    """Restores the Harmony tags that decoding a GPT-OSS output without special tokens removes."""
    if not text.startswith("analysis"):
        return text
    text = HARMONY_ANALYSIS + text[len("analysis"):]
    for plain, tagged in HARMONY_TURNS.items():
        text = text.replace(plain, tagged)
    return text


def load_outputs(tr_data_dir, model_type: ModelType | None = None) -> dict[int, list[str]]:
    """Outputs per question. Given ModelType.GPT_OSS, the outputs regain their Harmony tags."""
    texts = torch.load(Path(tr_data_dir) / "test_generated_texts.pt", map_location="cpu")
    if model_type == ModelType.GPT_OSS:
        texts = {g: [harmony_output(t) for t in v] for g, v in texts.items()}
    return {int(g): list(v) for g, v in texts.items()}


def load_thoughts(tr_data_dir, families, hidden_size: int, logger) -> dict[str, dict[int, torch.Tensor]]:
    out = {}
    for family in families:
        loader = ThoughtRepresentationLoader(
            logger=logger,
            tr_type=ThoughtRepresentation(family),
            cached_data_dir=str(tr_data_dir),
            split_name="test",
            vec_dim=hidden_size,
        )
        out[family] = {int(g): v for g, v in loader.thought_vecs_cache.items()}
    return out


ATTENTION_ROWS = 1024


def gpt_oss_eager_rows(module, query, key, value, attention_mask, **kwargs):
    parts = []
    for i in range(0, query.size(2), ATTENTION_ROWS):
        mask = None if attention_mask is None else attention_mask[:, :, i : i + ATTENTION_ROWS]
        parts.append(gpt_oss_eager(module, query[:, :, i : i + ATTENTION_ROWS], key, value, mask, **kwargs)[0])
    return torch.cat(parts, dim=1), None


AttentionInterface.register("gpt_oss_eager_rows", gpt_oss_eager_rows)
AttentionMaskInterface.register("gpt_oss_eager_rows", eager_mask)


def attention_backend(model_type) -> str:
    return "gpt_oss_eager_rows" if model_type == ModelType.GPT_OSS else "sdpa"


def load_model(base_llm_cfg, logger) -> BaseLLM:
    llm = BaseLLM.from_config(base_llm_cfg, logger)
    llm.model_kwargs["attn_implementation"] = attention_backend(llm.model_type)
    llm.load()
    return llm


def prompt_text(llm: BaseLLM, question: str) -> str:
    return llm._get_formatted([question], instruction=BBEHLoader.INSTRUCTION)[0]


def encode_first(tokenizer, text: str) -> list[int]:
    return tokenizer(text).input_ids


def encode_rest(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def embedding_norm(model) -> float:
    weight = model.get_input_embeddings().weight
    return float(weight.float().norm(dim=1).mean())


def random_vectors(length: int, dim: int, norm: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    vecs = torch.randn((length, dim), generator=gen)
    return vecs * (norm / vecs.norm(dim=1, keepdim=True).clamp_min(1e-12))


def model_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def model_dtype(model) -> torch.dtype:
    return model.get_input_embeddings().weight.dtype


def embed_ids(model, ids) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long, device=model_device(model))
    return model.get_input_embeddings()(ids)


def sliding_window(model):
    layer_types = getattr(model.config, "layer_types", None) or []
    return getattr(model.config, "sliding_window", None) if "sliding_attention" in layer_types else None


def packed_mask(positions: torch.Tensor, segments: torch.Tensor, window, dtype, device) -> torch.Tensor:
    q, k = torch.arange(len(positions))[:, None], torch.arange(len(positions))[None, :]
    allowed = (k <= q) & ((segments[None, :] <= 1) | (segments[None, :] == segments[:, None]))
    if window is not None:
        allowed &= (positions[:, None] - positions[None, :]) < window
    mask = torch.zeros(allowed.shape, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None].to(device)


def packed_hidden(model, prefix: torch.Tensor, lead, blocks: list[list[int]]) -> list[torch.Tensor]:
    device, dtype = model_device(model), model_dtype(model)
    batch = 1 if lead is None else lead.size(0)
    lead_len = 0 if lead is None else lead.size(1)
    base = prefix.size(0) + lead_len
    parts = [prefix.to(device, dtype).unsqueeze(0).expand(batch, -1, -1)]
    positions = [torch.arange(base)]
    segments = [torch.zeros(prefix.size(0), dtype=torch.long), torch.ones(lead_len, dtype=torch.long)]
    if lead_len:
        parts.append(lead.to(device, dtype))
    starts, cursor = [], base
    for j, block in enumerate(blocks):
        width = len(block) - 1
        starts.append(cursor)
        if width > 0:
            parts.append(embed_ids(model, block[:-1]).unsqueeze(0).expand(batch, -1, -1))
            positions.append(torch.arange(base, base + width))
            segments.append(torch.full((width,), j + 2, dtype=torch.long))
        cursor += width
    positions, segments = torch.cat(positions), torch.cat(segments)
    full = packed_mask(positions, segments, None, dtype, device)
    window = sliding_window(model)
    mask = full if window is None else {
        "full_attention": full,
        "sliding_attention": packed_mask(positions, segments, window, dtype, device),
    }
    hidden = model.base_model(
        inputs_embeds=torch.cat(parts, dim=1),
        attention_mask=mask,
        position_ids=positions[None].expand(batch, -1).to(device),
        use_cache=False,
    ).last_hidden_state
    out = []
    for start, block in zip(starts, blocks):
        index = torch.cat([torch.tensor([base - 1]), torch.arange(start, start + len(block) - 1)]).to(hidden.device)
        out.append(hidden[:, index])
    return out


def chunk_blocks(sizes: list[int], fixed: int, cap: int) -> list[list[int]]:
    chunks, current, used = [], [], fixed
    for i, size in enumerate(sizes):
        need = max(size - 1, 0)
        if current and used + need > cap:
            chunks.append(current)
            current, used = [], fixed
        current.append(i)
        used += need
    if current:
        chunks.append(current)
    return chunks


def token_log_probs(model, hidden: torch.Tensor, chunk: int = 256):
    head = model.get_output_embeddings()
    for start in range(0, hidden.size(-2), chunk):
        yield start, torch.log_softmax(head(hidden[..., start : start + chunk, :]).float(), dim=-1)


def token_ce(model, hidden: torch.Tensor, targets: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    parts = []
    for start, lp in token_log_probs(model, hidden, chunk):
        tgt = targets[..., start : start + lp.size(-2)].to(lp.device)
        parts.append(-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
    return torch.cat(parts, dim=-1)


def ce_and_kl(model, hidden: torch.Tensor, targets: torch.Tensor, ref_log_probs: torch.Tensor, chunk: int = 128):
    ce, kl = [], []
    for start, lq in token_log_probs(model, hidden, chunk):
        width = lq.size(-2)
        lp = ref_log_probs[start : start + width].to(lq.device)
        tgt = targets[start : start + width].to(lq.device).expand(lq.shape[:-1])
        ce.append(-lq.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
        kl.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(ce, dim=-1), torch.cat(kl, dim=-1)


def explicit_ce_and_tail(model, hidden: torch.Tensor, targets: torch.Tensor, tail: int, chunk: int = 256):
    m = hidden.size(-2)
    ce, tail_parts = [], []
    for start, lp in token_log_probs(model, hidden, chunk):
        tgt = targets[start : start + lp.size(-2)].to(lp.device)
        ce.append(-lp[0].gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
        first = max(start, m - tail)
        if first < start + lp.size(-2):
            tail_parts.append(lp[0, first - start :])
    return torch.cat(ce), torch.cat(tail_parts) if tail_parts else None


def bootstrap_draws(per_question_sums: np.ndarray, per_question_counts: np.ndarray, resamples: int, seed: int):
    keep = per_question_counts > 0
    s, c = per_question_sums[keep], per_question_counts[keep]
    ids = np.random.default_rng(seed).integers(0, len(c), (resamples, len(c)))
    return s.sum() / c.sum(), s[ids].sum(1) / c[ids].sum(1)


def bootstrap_mean(per_question_sums: np.ndarray, per_question_counts: np.ndarray, resamples: int, seed: int):
    if not (per_question_counts > 0).any():
        return None, None
    mean, boot = bootstrap_draws(per_question_sums, per_question_counts, resamples, seed)
    return float(mean), float(np.std(boot, ddof=1))


def interval(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def recovered_fractions(sums: dict[str, np.ndarray], counts: np.ndarray, resamples: int, seed: int) -> dict:
    base, base_draws = bootstrap_draws(sums["question|kl"], counts, resamples, seed)
    out = {}
    for name, s in sums.items():
        arm, key = name.split("|")
        if key == "kl":
            point, draws = bootstrap_draws(s, counts, resamples, seed)
            out[arm] = dict(recovered=float(1 - point / base), recovered_ci=interval(1 - draws / base_draws))
    return out


def information_ratios(sums: dict[str, np.ndarray], counts: np.ndarray, resamples: int, seed: int) -> dict:
    draw = {name: bootstrap_draws(s, counts, resamples, seed) for name, s in sums.items()}
    (ey, ey_d), (oy, oy_d) = draw["empty|ce_y"], draw["output|ce_y"]
    (ex, ex_d), (qx, qx_d) = draw["empty|ce_x"], draw["question|ce_x"]
    out = {}
    for arm in {name.split("|")[0] for name in sums}:
        (vy, vy_d), (vx, vx_d) = draw[f"{arm}|ce_y"], draw[f"{arm}|ce_x"]
        out[arm] = dict(r_y=float((ey - vy) / (ey - oy)), r_y_ci=interval((ey_d - vy_d) / (ey_d - oy_d)),
                        r_x=float((ex - vx) / (ex - qx)), r_x_ci=interval((ex_d - vx_d) / (ex_d - qx_d)))
    return out


def fold_instruction(llm: BaseLLM) -> bool:
    return llm.model_type in (ModelType.QWEN, ModelType.DEEPSEEK)
