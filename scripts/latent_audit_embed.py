import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

from src.model.latent_methods import local

EMBEDDER = "nvidia/llama-embed-nemotron-8b"


class StreamedEmbedder:

    def __init__(self, repo: str = EMBEDDER, max_length: int = 4096, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.root = Path(local(repo))
        self.config = LlamaConfig.from_pretrained(self.root)
        self.config._attn_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(self.root)
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.device, self.dtype = torch.device(device), dtype
        self.rotary = LlamaRotaryEmbedding(self.config).to(self.device)
        self.max_length = max_length

    def tensors(self, prefix: str) -> dict[str, torch.Tensor]:
        out = {}
        for file in sorted({f for k, f in self.weight_map.items() if k.startswith(prefix)}):
            with safe_open(self.root / file, "pt") as fh:
                for key in fh.keys():
                    if key.startswith(prefix):
                        out[key[len(prefix) :]] = fh.get_tensor(key).to(self.device, self.dtype)
        return out

    def layer(self, i: int) -> LlamaDecoderLayer:
        with torch.device("meta"):
            layer = LlamaDecoderLayer(self.config, layer_idx=i)
        layer.load_state_dict(self.tensors(f"layers.{i}."), assign=True)
        layer.self_attn.is_causal = False
        return layer.eval()

    def batches(self, ids: list[list[int]], batch_tokens: int) -> list[list[int]]:
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out, current = [], []
        for i in order:
            if current and len(ids[i]) * (len(current) + 1) > batch_tokens:
                out.append(current)
                current = []
            current.append(i)
        if current:
            out.append(current)
        return out

    def embed_chunk(self, ids: list[list[int]], batch_tokens: int, log) -> torch.Tensor:
        embed = self.tensors("embed_tokens.")["weight"]
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        state = []
        for members in self.batches(ids, batch_tokens):
            width = max(len(ids[i]) for i in members)
            rows = torch.full((len(members), width), pad, dtype=torch.long)
            keep = torch.zeros((len(members), width), dtype=torch.bool)
            for r, i in enumerate(members):
                rows[r, width - len(ids[i]) :] = torch.as_tensor(ids[i])
                keep[r, width - len(ids[i]) :] = True
            rows, keep = rows.to(self.device), keep.to(self.device)
            mask = torch.zeros((len(members), 1, width, width), dtype=self.dtype, device=self.device)
            mask.masked_fill_(~keep[:, None, None, :], torch.finfo(self.dtype).min)
            positions = torch.arange(width, device=self.device)[None].expand(len(members), -1)
            state.append(dict(members=members, hidden=embed[rows], keep=keep, mask=mask, positions=positions,
                              rope=self.rotary(embed[:1][None], positions)))
        del embed
        for i in range(self.config.num_hidden_layers):
            t0 = time.time()
            layer = self.layer(i)
            with torch.no_grad():
                for s in state:
                    out = layer(s["hidden"], attention_mask=s["mask"], position_ids=s["positions"],
                                position_embeddings=s["rope"])
                    s["hidden"] = out[0] if isinstance(out, tuple) else out
            del layer
            log(f"layer {i} {time.time() - t0:.1f}s")
        norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.load_state_dict(self.tensors("norm."), assign=True)
        result = torch.empty((len(ids), self.config.hidden_size))
        with torch.no_grad():
            for s in state:
                hidden = norm(s["hidden"]).float().masked_fill(~s["keep"][..., None], 0.0)
                pooled = hidden.sum(1) / s["keep"].sum(1, keepdim=True)
                result[s["members"]] = F.normalize(pooled, dim=-1).cpu()
        return result

    def embed(self, texts: list[str], chunk_tokens: int, batch_tokens: int, log=print) -> torch.Tensor:
        ids = [self.tokenizer(t, max_length=self.max_length, truncation=True).input_ids for t in texts]
        chunks, current, used = [], [], 0
        for i, row in enumerate(ids):
            if current and used + len(row) > chunk_tokens:
                chunks.append(current)
                current, used = [], 0
            current.append(i)
            used += len(row)
        if current:
            chunks.append(current)
        log(f"{len(texts)} texts, {sum(map(len, ids))} tokens, {len(chunks)} chunks")
        out = torch.empty((len(texts), self.config.hidden_size))
        for c, members in enumerate(chunks):
            log(f"chunk {c + 1}/{len(chunks)}")
            out[members] = self.embed_chunk([ids[i] for i in members], batch_tokens, log)
        return out


def main() -> None:
    p = argparse.ArgumentParser(description="Nemotron embeddings of the questions and explicit outputs, one layer in memory at a time.")
    p.add_argument("--data-dir", type=Path, default=Path("outputs/latent_audit"))
    p.add_argument("--methods", nargs="+", default=["codi", "sim_cot", "colar", "llama"])
    p.add_argument("--chunk-tokens", type=int, default=150_000)
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                   help="default: bfloat16 on GPU as in the main pipeline, float32 on CPU")
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype or ("float32" if args.device == "cpu" else "bfloat16"))
    records = {m: torch.load(args.data_dir / f"{m}.pt", weights_only=False) for m in args.methods}
    questions = next(iter(records.values()))["questions"]
    if any(r["questions"] != questions for r in records.values()):
        raise ValueError("methods hold different questions")
    texts, index = [], {}
    for text in questions + [t for r in records.values() for row in r["outputs"] for t in row]:
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
    embedder = StreamedEmbedder(device=args.device, dtype=dtype)
    vecs = embedder.embed(texts, args.chunk_tokens, args.batch_tokens, log=lambda s: print(s, flush=True))
    out = dict(model=EMBEDDER, questions=vecs[[index[q] for q in questions]])
    for m, r in records.items():
        out[m] = torch.stack([vecs[[index[t] for t in row]] for row in r["outputs"]])
    torch.save(out, args.data_dir / "embeddings.pt")


if __name__ == "__main__":
    main()
