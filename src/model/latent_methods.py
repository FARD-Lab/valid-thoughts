import importlib.util
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from src.dataset.bbeh_loader import BBEHLoader
from src.model.llm import BaseLLM
from src.utils.config import ModelType

BASE = "meta-llama/Llama-3.2-1B-Instruct"
GPT2 = "openai-community/gpt2"
LORA_SCALE = 32 / 128
ANSWER_PREFIX = "The answer is:"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"


def local(repo: str, filename: str | None = None, repo_type: str | None = None) -> str:
    if filename is None:
        return snapshot_download(repo, repo_type=repo_type, local_files_only=True)
    return hf_hub_download(repo, filename, repo_type=repo_type, local_files_only=True)


def gsm8k(split: str = "test") -> list[dict]:
    path = local("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(path).to_pylist()


def to_number(text: str):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def gold_answer(answer: str):
    return to_number(answer.split("####")[-1].strip())


def extract_answer(text: str):
    for marker in (ANSWER_PREFIX, "Answer:"):
        if marker in text:
            found = NUMBER.findall(text.rsplit(marker, 1)[1])
            if found:
                return to_number(found[0])
    found = NUMBER.findall(text)
    return to_number(found[-1]) if found else None


def find_last(ids: list[int], pattern: list[int]):
    for start in range(len(ids) - len(pattern), -1, -1):
        if ids[start : start + len(pattern)] == pattern:
            return start
    return None


def trim(ids: list[int], stops: set[int], pad: int) -> list[int]:
    out = []
    for token in ids:
        if token in stops:
            out.append(token)
            break
        if token != pad:
            out.append(token)
    return out


def third_party_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, THIRD_PARTY / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_lora(model: nn.Module, state: dict, prefix: str, scale: float, transpose: bool = False) -> None:
    """Merge the LoRA adapters of state into the weights. GPT-2 stores its linear layers as Conv1D, whose weight is
    the transpose of an nn.Linear weight, so its adapters merge with transpose=True."""
    params = dict(model.named_parameters())
    for key in state:
        if key.startswith(prefix) and key.endswith(".lora_A.default.weight"):
            stem = key[: -len(".lora_A.default.weight")]
            delta = state[f"{stem}.lora_B.default.weight"].float() @ state[key].float()
            params[f"{stem[len(prefix):]}.weight"].data += scale * (delta.T if transpose else delta)


def base_model(vocab: int, base: str = BASE) -> nn.Module:
    if base == BASE:
        model = LlamaForCausalLM.from_pretrained(local(BASE), dtype=torch.float32, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(local(base), dtype=torch.float32, attn_implementation="sdpa")
    old = model.get_input_embeddings().weight.shape[0]
    if vocab != old:
        model.resize_token_embeddings(vocab, mean_resizing=False)
        with torch.no_grad():
            weight = model.get_input_embeddings().weight
            weight[old:] = weight[:old].mean(0)
    return model.eval()


class Method:
    name: str
    families: list[str]
    open_ids: list[int] = []
    close_ids: list[int] = []
    post_ids: list[int] = []
    max_new_tokens = 256
    sampling: dict = {}
    chat_template = True

    model: nn.Module
    tokenizer = None
    banned: list[int] = []

    @property
    def stops(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        return set(eos if isinstance(eos, list) else [eos])

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.stops)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def to(self, device) -> "Method":
        self.model.to(device)
        return self

    def embed(self, ids) -> torch.Tensor:
        return self.model.get_input_embeddings()(torch.as_tensor(ids, dtype=torch.long, device=self.device))

    def last_hidden(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.model.base_model(inputs_embeds=embeds[None]).last_hidden_state[0, -1]

    def generate(self, embeds: torch.Tensor, suppress: list[int], **kwargs) -> list[int]:
        return self.generate_many(embeds, suppress, 1, **kwargs)[0]

    def generate_many(self, embeds: torch.Tensor, suppress: list[int], n: int, **kwargs) -> list[list[int]]:
        mask = torch.ones(1, embeds.size(0), dtype=torch.long, device=self.device)
        out = self.model.generate(inputs_embeds=embeds[None], attention_mask=mask, pad_token_id=self.pad_id,
                                  suppress_tokens=suppress or None, num_return_sequences=n, **kwargs)
        return [trim(row.tolist(), self.stops, self.pad_id) for row in out]

    def answer_embeds(self, question: str, thought: torch.Tensor) -> torch.Tensor:
        """The latent prompt, the thought, and the tokens after it, from which the method decodes its answer."""
        return torch.cat([self.embed(self.latent_prompt(question)), thought.to(self.device, torch.float32),
                          self.embed(self.post_ids)])

    def sample_explicit(self, question: str, n: int, seed: int) -> list[list[int]]:
        torch.manual_seed(seed)
        ids = torch.tensor([self.explicit_prompt(question)], device=self.device)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), num_return_sequences=n,
                                  max_new_tokens=self.max_new_tokens, pad_token_id=self.pad_id,
                                  suppress_tokens=self.banned or None, **self.sampling)
        return [trim(row[ids.size(1) :].tolist(), self.stops, self.pad_id) for row in out]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def explicit_prompt(self, question: str) -> list[int]:
        raise NotImplementedError

    def latent_prompt(self, question: str) -> list[int]:
        raise NotImplementedError

    def thoughts(self, question: str, seed: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def latent_answer(self, question: str, thought: torch.Tensor, seed: int):
        return None

    def answer_start(self, ids: list[int]):
        raise NotImplementedError


class Codi(Method):
    num_latent = 6
    answer_tokens = 64

    def __init__(self, name: str, repo: str, filename: str, base: str = BASE):
        self.name = name
        self.families = [name]
        path = local(repo, filename)
        if path.endswith(".safetensors"):
            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.tokenizer = AutoTokenizer.from_pretrained(local(base))
        vocab = len(self.tokenizer)
        self.bot_id, self.eot_id = vocab + 1, vocab + 2
        self.codi_pad = vocab
        self.model = base_model(vocab + 3, base)
        prefix = "codi.base_model.model."
        plain = {k[len(prefix) :].replace(".base_layer", ""): v for k, v in state.items()
                 if k.startswith(prefix) and ".lora_" not in k}
        missing, unexpected = self.model.load_state_dict(plain, strict=False)
        if unexpected or [k for k in missing if k != "lm_head.weight"]:
            raise ValueError(f"{name}: missing {missing}, unexpected {unexpected}")
        add_lora(self.model, state, prefix, LORA_SCALE, transpose=base == GPT2)
        dim = self.model.config.hidden_size
        self.prj = nn.Sequential(nn.Dropout(0.0), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.prj.add_module("ln", nn.LayerNorm(dim))
        self.prj.load_state_dict({k[len("prj.") :]: v.float() for k, v in state.items() if k.startswith("prj.")})
        self.prj.eval()
        self.open_ids, self.close_ids, self.post_ids = [self.bot_id], [self.eot_id], [self.eot_id]
        self.banned = [self.codi_pad, self.bot_id, self.eot_id]
        self.answer_ids = self.tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)

    @property
    def pad_id(self) -> int:
        return self.codi_pad

    def to(self, device) -> "Codi":
        self.model.to(device)
        self.prj.to(device)
        return self

    def question_ids(self, question: str) -> list[int]:
        return self.tokenizer(question.strip().replace("  ", " ")).input_ids

    def explicit_prompt(self, question: str) -> list[int]:
        return self.question_ids(question)

    def latent_prompt(self, question: str) -> list[int]:
        return self.question_ids(question) + [self.bot_id]

    def thoughts(self, question: str, seed: int) -> dict[str, torch.Tensor]:
        embeds = self.embed(self.latent_prompt(question))
        latents = []
        z = self.prj(self.last_hidden(embeds))
        for _ in range(self.num_latent):
            latents.append(z)
            embeds = torch.cat([embeds, z[None]])
            z = self.prj(self.last_hidden(embeds))
        return {self.name: torch.stack(latents)}

    @property
    def answer_suppress(self) -> list[int]:
        return [self.eot_id]

    def latent_answer(self, question: str, thought: torch.Tensor, seed: int):
        ids = self.generate(self.answer_embeds(question, thought), self.answer_suppress, do_sample=False,
                            temperature=None, top_p=None, max_new_tokens=self.answer_tokens)
        return self.decode(ids)

    def answer_start(self, ids: list[int]):
        return find_last(ids, self.answer_ids)


class CodiGpt2(Codi):
    """CODI and SIM-CoT on GPT-2 small. The GPT-2 tokenizer adds no BOS token and the checkpoints keep the training
    format of the CODI code in the SIM-CoT repository (question, bot, six latents, eot, "The answer is:", eos), so only the base
    model, the LoRA merge, and the prompt format of the audit prompts differ from the Llama checkpoints. GPT-2 has
    no chat template, so the minimality prompts are plain text, and its generation config does not sample, so the
    explicit outputs sample with the settings of the Llama-3.2-1B-Instruct generation config."""

    chat_template = False
    sampling = dict(do_sample=True, temperature=0.6, top_p=0.9)

    def __init__(self, name: str, repo: str, filename: str):
        super().__init__(name, repo, filename, GPT2)


class Colar(Method):
    name = "colar"
    families = ["colar", "colar_mean"]
    speed = 5
    max_latents = 64
    answer_tokens = 16
    template = "Question: {} Let's think step by step:(Thinking speed: {})###"

    def __init__(self):
        path = local("AlbertTan/CoLaR", "logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt")
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        self.tokenizer = AutoTokenizer.from_pretrained(local(BASE))
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = base_model(len(self.tokenizer))
        add_lora(self.model, state, "llm.base_model.model.", LORA_SCALE)
        policy = third_party_module("colar_projector", "colar/src/modules/projector.py").LatentPolicy
        dim = self.model.config.hidden_size
        self.policy = policy(dim, intermediate_size=dim, deterministic=False)
        self.policy.load_state_dict({k[len("latent_policy.") :]: v.float() for k, v in state.items()
                                     if k.startswith("latent_policy.")})
        self.policy.eval()
        std = third_party_module("colar_constants", "colar/src/utils/constants.py").MODEL_EMB_STD
        self.embeds_std = std["Llama-3.2-1B-Instruct"]
        self.sep_id = self.tokenizer.convert_tokens_to_ids("###")
        self.open_ids, self.close_ids, self.post_ids = [self.sep_id], [self.sep_id], [self.sep_id]
        self.banned = [self.tokenizer.pad_token_id]

    def to(self, device) -> "Colar":
        self.model.to(device)
        self.policy.to(device)
        return self

    def prompt(self, question: str, speed) -> list[int]:
        return self.tokenizer(self.template.format(question, speed), add_special_tokens=False).input_ids

    def explicit_prompt(self, question: str) -> list[int]:
        return self.prompt(question, 1)

    def latent_prompt(self, question: str) -> list[int]:
        return self.prompt(question, self.speed)

    def latent_loop(self, question: str, seed: int, sample: bool) -> torch.Tensor:
        torch.manual_seed(seed)
        embeds = self.embed(self.latent_prompt(question))
        hidden = self.last_hidden(embeds)
        latents = []
        for _ in range(self.max_latents):
            dist = self.policy(hidden[None])
            z = (dist.rsample() if sample else dist.mean)[0] * self.embeds_std
            latents.append(z)
            embeds = torch.cat([embeds, z[None]])
            hidden = self.last_hidden(embeds)
            probs = torch.softmax(self.model.lm_head(hidden).float(), dim=-1)
            if int(torch.multinomial(probs, 1)) == self.sep_id:
                break
        return torch.stack(latents)

    def thoughts(self, question: str, seed: int) -> dict[str, torch.Tensor]:
        return {"colar": self.latent_loop(question, seed, True), "colar_mean": self.latent_loop(question, seed, False)}

    @property
    def answer_suppress(self) -> list[int]:
        return self.banned

    def latent_answer(self, question: str, thought: torch.Tensor, seed: int):
        torch.manual_seed(seed)
        ids = self.generate(self.answer_embeds(question, thought), self.answer_suppress, do_sample=True, top_p=0.9,
                            temperature=1.0, max_new_tokens=self.answer_tokens)
        return self.decode(ids)

    def answer_start(self, ids: list[int]):
        if self.sep_id not in ids:
            return None
        return len(ids) - ids[::-1].index(self.sep_id)


class TrainingFree(Method):
    name = "llama"
    families = ["soft_thinking", "soft_thinking_noise", "latent_thinking"]
    steps = 128
    max_new_tokens = 768

    def __init__(self, logger, device: str = "cpu"):
        self.llm = BaseLLM(logger=logger, model_name=BASE, device=device, torch_dtype="float32",
                           model_type=ModelType.LLAMA)
        self.llm.model_kwargs["attn_implementation"] = "sdpa"
        self.llm.load()
        self.model, self.tokenizer = self.llm.model, self.llm.tokenizer
        self.answer_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in (ANSWER_PREFIX, " " + ANSWER_PREFIX)]

    def to(self, device) -> "TrainingFree":
        return self

    def explicit_prompt(self, question: str) -> list[int]:
        text = self.llm._get_formatted([question], instruction=BBEHLoader.INSTRUCTION)[0]
        return self.tokenizer(text).input_ids

    def latent_prompt(self, question: str) -> list[int]:
        return self.explicit_prompt(question)

    def thoughts(self, question: str, seed: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        vecs = self.llm.get_combined_thinking(question, BBEHLoader.INSTRUCTION, steps=self.steps)
        return {family: v[0].float() for family, v in zip(self.families, vecs)}

    def answer_start(self, ids: list[int]):
        found = [s for s in (find_last(ids, p) for p in self.answer_ids) if s is not None]
        return max(found) if found else None


def load_method(name: str, logger=None, device: str = "cpu") -> Method:
    if name == "codi":
        method = Codi("codi", "zen-E/CODI-llama3.2-1b-Instruct", "pytorch_model.bin")
    elif name == "sim_cot":
        method = Codi("sim_cot", "internlm/SIM_COT-LLaMA3-CODI-1B", "model.safetensors")
    elif name == "codi_gpt2":
        method = CodiGpt2("codi_gpt2", "zen-E/CODI-gpt2", "pytorch_model.bin")
    elif name == "sim_cot_gpt2":
        method = CodiGpt2("sim_cot_gpt2", "internlm/SIM_COT-GPT2-CODI", "model-00001-of-00001.safetensors")
    elif name == "colar":
        method = Colar()
    elif name == "llama":
        method = TrainingFree(logger, device)
    else:
        raise ValueError(f"unknown method {name}")
    return method.to(device)
