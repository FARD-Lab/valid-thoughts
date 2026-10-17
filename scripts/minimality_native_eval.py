import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from scripts.native_common import (
    HARMONY_FINAL,
    bootstrap_mean,
    chunk_blocks,
    embed_ids,
    embedding_norm,
    encode_first,
    encode_rest,
    fold_instruction,
    information_ratios,
    load_model,
    load_outputs,
    load_questions,
    load_thoughts,
    packed_hidden,
    random_vectors,
    token_ce,
)
from src.dataset.bbeh_loader import BBEHLoader
from src.utils import set_seed_all
from src.utils.config import MinimalityNativeEvalConfig, ModelType, register_configs
from src.utils.logging import Logger

register_configs()

THOUGHT_SLOT = "<<THOUGHT>>"
OUTPUT_SLOT = "<<OUTPUT>>"
WRAPPER = "Here is the thought representation of a question you were asked: {thought}\nAnswer the question."
RESTATE = "Restate the question you answered."


def chat_segments(llm, date: str | None = None) -> tuple[str, str, str]:
    """The wrapper text before and after the thought and between the output and the question. A Llama chat template
    writes the date of the run into its system prompt, and date (e.g. "30 Sep 2026") replaces it."""
    wrapper = WRAPPER.format(thought=THOUGHT_SLOT)
    if fold_instruction(llm):
        system, user = [], f"{BBEHLoader.INSTRUCTION}\n\n{wrapper}"
    else:
        system, user = [{"role": "system", "content": BBEHLoader.INSTRUCTION}], wrapper
    pinned = {"date_string": date} if date else {}
    first = llm.tokenizer.apply_chat_template(
        system + [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, **pinned
    )
    if llm.model_type == ModelType.QWEN:
        first = first + "<think>\n"
    full = llm.tokenizer.apply_chat_template(
        system + [{"role": "user", "content": user}, {"role": "assistant", "content": OUTPUT_SLOT},
                  {"role": "user", "content": RESTATE}],
        add_generation_prompt=True,
        tokenize=False,
    )
    before, after = first.split(THOUGHT_SLOT)
    between = full.split(OUTPUT_SLOT, 1)[1]
    if llm.model_type == ModelType.GPT_OSS:
        between = between + HARMONY_FINAL
    return before, after, between


def plain_segments() -> tuple[str, str, str]:
    """The same segments as chat_segments for a model without a chat template: the instruction and the wrapper as
    plain text around the thought slot, then the output, and the restate request before the question."""
    before, after = f"{BBEHLoader.INSTRUCTION}\n\n{WRAPPER.format(thought=THOUGHT_SLOT)}".split(THOUGHT_SLOT)
    return before, after + "\n\n", f"\n\n{RESTATE}\n\n"


def score_blocks(model, prefix, lead, blocks, spans):
    hidden = packed_hidden(model, prefix, lead, blocks)
    out = []
    for h, block, (y_span, x_span) in zip(hidden, blocks, spans):
        targets = torch.as_tensor(block).expand(h.size(0), -1)
        ce = token_ce(model, h, targets)
        out.append((ce[:, y_span[0] : y_span[1]], ce[:, x_span[0] : x_span[1]]))
    return out


def to_half(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float16)


def summarize(records, groups, resamples, seed):
    index = {g: i for i, g in enumerate(groups)}
    sums: dict[str, np.ndarray] = {}
    counts = np.zeros(len(groups))
    for r in records:
        i = index[r["g_idx"]]
        counts[i] += 1
        for arm, vals in r["arms"].items():
            ce_y = float(np.mean(vals["ce_y"], dtype=np.float64))
            ce_x = float(np.mean(vals["ce_x"], dtype=np.float64))
            for key, value in (("ce_y", ce_y), ("ce_x", ce_x), ("delta_ib", ce_x - ce_y)):
                sums.setdefault(f"{arm}|{key}", np.zeros(len(groups)))[i] += value
    out = {}
    for name, s in sums.items():
        arm, key = name.split("|")
        mean, se = bootstrap_mean(s, counts, resamples, seed)
        out.setdefault(arm, {})[key] = mean
        out[arm][f"{key}_se"] = se
    if counts.any() and {"empty|ce_y", "output|ce_y", "question|ce_x"} <= sums.keys():
        for arm, vals in information_ratios(sums, counts, resamples, seed).items():
            out[arm].update(vals)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="minimality_native_eval")
def main(cfg: MinimalityNativeEvalConfig) -> None:
    logger = Logger.from_config(cfg.logging)
    logger.info("Minimality Native Eval")
    logger.info(OmegaConf.to_yaml(cfg))
    set_seed_all(cfg.seed)

    llm = load_model(cfg.base_llm, logger)
    model, tokenizer = llm.model, llm.tokenizer
    hidden_size = model.config.hidden_size
    norm = embedding_norm(model)

    families = list(cfg.families)
    steps = sorted(int(s) for s in cfg.steps)
    questions = load_questions(cfg.llm_data_dir)
    # The output arm reads the output text inside the user message, where Harmony tags would end the message,
    # so only the targets regain their tags.
    outputs = load_outputs(cfg.tr_data_dir)
    targets = load_outputs(cfg.tr_data_dir, llm.model_type)
    groups = sorted(outputs)
    thoughts = load_thoughts(cfg.tr_data_dir, families, hidden_size, logger)
    eval_groups = groups[: cfg.num_examples] if cfg.num_examples > 0 else groups
    before, after, between = chat_segments(llm)
    pre_ids, post_ids, mid_ids = encode_first(tokenizer, before), encode_rest(tokenizer, after), encode_rest(tokenizer, between)
    prefix = embed_ids(model, pre_ids)
    cap = cfg.max_row_tokens

    records = []
    with torch.no_grad():
        for g in tqdm(eval_groups, desc="minimality native", unit="question"):
            x_ids = encode_rest(tokenizer, questions[g])
            rv = random_vectors(steps[-1], hidden_size, norm, cfg.seed + g)
            n_beams = cfg.beams if cfg.beams > 0 else len(outputs[g])
            slots = [encode_rest(tokenizer, t) for t in outputs[g][:n_beams]]
            ys = [encode_rest(tokenizer, t) for t in targets[g][:n_beams]]
            beams = [b for b, y in enumerate(ys) if y and x_ids]
            recs = {b: dict(g_idx=g, beam_idx=b, n_y=len(ys[b]), n_x=len(x_ids), arms={}) for b in beams}
            blocks = {b: post_ids + ys[b] + mid_ids + x_ids for b in beams}
            spans = {}
            for b in beams:
                y0 = len(post_ids)
                x0 = y0 + len(ys[b]) + len(mid_ids)
                spans[b] = ((y0, y0 + len(ys[b])), (x0, x0 + len(x_ids)))

            def run(lead, names, members, fixed):
                sizes = [len(blocks[b]) for b in members]
                for chunk in chunk_blocks(sizes, fixed, cap):
                    chosen = [members[i] for i in chunk]
                    scored = score_blocks(model, prefix, lead, [blocks[b] for b in chosen], [spans[b] for b in chosen])
                    for b, (ce_y, ce_x) in zip(chosen, scored):
                        for a, name in enumerate(names):
                            recs[b]["arms"][name] = dict(ce_y=to_half(ce_y[a]), ce_x=to_half(ce_x[a]))

            run(embed_ids(model, x_ids).unsqueeze(0), ["question"], beams, len(pre_ids) + len(x_ids))
            run(None, ["empty"], beams, len(pre_ids))
            for b in beams:
                run(embed_ids(model, slots[b]).unsqueeze(0), ["output"], [b], len(pre_ids) + len(slots[b]))
            for s in steps:
                names = [f"{family}_{s}" for family in families] + ([f"random_{s}"] if cfg.random_vector else [])
                vecs = [thoughts[family][g][:s].float() for family in families]
                if cfg.random_vector:
                    vecs.append(rv[:s])
                run(torch.stack(vecs), names, beams, len(pre_ids) + s)
            records.extend(recs[b] for b in beams)

    summary = summarize(records, eval_groups, cfg.bootstrap_resamples, cfg.seed)
    for arm, vals in summary.items():
        logger.info(f"{arm:28s} CE(Y|T) {vals['ce_y']:.3f}  CE(X|Y,T) {vals['ce_x']:.3f}  delta_IB {vals['delta_ib']:.3f}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(model=cfg.model_label or cfg.base_llm.name, families=families, steps=steps, beams=cfg.beams,
                seed=cfg.seed, wrapper=WRAPPER, restate=RESTATE,
                bootstrap=dict(resamples=cfg.bootstrap_resamples, seed=cfg.seed))
    (out_dir / "minimality_native_summary.json").write_text(
        json.dumps(meta | dict(n_questions=len({r["g_idx"] for r in records}), n_outputs=len(records), arms=summary), indent=1))
    torch.save(dict(meta=meta, records=records), out_dir / "minimality_native.pt")
    logger.info(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
