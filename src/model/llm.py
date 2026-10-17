import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import convert_file_size_to_int
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.model.base import BaseModel
from src.utils.config import BaseLLMConfig, ModelType
from src.utils.logging import Logger

def strip_thinking_tokens(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _is_enabled(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def offload_device_map(skeleton, max_memory: Dict[Union[int, str], Union[int, str]]) -> Dict[str, Union[int, str]]:
    budget = {k: convert_file_size_to_int(v) if isinstance(v, str) else v for k, v in max_memory.items()}
    gpu = min(k for k in budget if isinstance(k, int))
    budget[gpu] -= sum(p.numel() * p.element_size() for p in skeleton.get_output_embeddings().parameters())
    device_map = infer_auto_device_map(skeleton, max_memory=budget, no_split_module_classes=skeleton._no_split_modules)
    return {k: v if ".layers." in f".{k}." else gpu for k, v in device_map.items()}

class BaseLLM(BaseModel):
    def __init__(
        self,
        logger: Logger,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = "auto",
        trust_remote_code: bool = False,
        max_memory: Optional[Dict[str, str]] = None,
        model_type: Optional[ModelType] = None,
        **kwargs,
    ):
        super().__init__(
            logger=logger,
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            model_type=model_type,
            **kwargs,
        )
        self.model_type = model_type
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def from_config(config: BaseLLMConfig, logger: Logger) -> "BaseLLM":
        def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        model_type_value = _cfg_get(config, "model_type", None)
        if isinstance(model_type_value, str):
            model_type_value = ModelType(model_type_value.lower())

        return BaseLLM(
            logger=logger,
            model_name=_cfg_get(config, "name", ""),
            device=_cfg_get(config, "device", "cuda"),
            torch_dtype=_cfg_get(config, "torch_dtype", "auto"),
            trust_remote_code=_cfg_get(config, "trust_remote_code", False),
            max_memory=_cfg_get(config, "max_memory", None),
            model_type=model_type_value,
        )

    def load(self, tokenizer_only: bool = False) -> None:
        self.validate_offline_cache()

        self.logger.info(f"Loading tokenizer for model '{self.model_name}'")
        offline_mode = self._offline_mode_enabled()
        cache_dir = self._resolve_hf_hub_cache_dir()
        common_load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.model_kwargs.get("trust_remote_code", False),
            "local_files_only": offline_mode,
        }
        if cache_dir:
            common_load_kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            **common_load_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if tokenizer_only:
            return

        self.logger.info(
            f"Model load start: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                self.logger.info(
                    "Detected CUDA device[0]: "
                    f"name={props.name}, total_memory={props.total_memory / (1024 ** 3):.1f}GiB"
                )
            except Exception as exc:
                self.logger.warning(f"Could not read CUDA device properties: {exc}")

        if self.max_memory:
            actual = torch.cuda.device_count()
            bad = [k for k in self.max_memory if isinstance(k, int) and k >= actual]
            if bad:
                raise RuntimeError(
                    f"max_memory specifies CUDA device(s) {bad} but only {actual} "
                    f"device(s) are available "
                    f"(CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')}). "
                    "A GPU on this node may be in ERR! state. Check nvidia-smi."
                )

        self.logger.info(f"Loading model '{self.model_name}'")
        load_start = time.monotonic()
        model_load_kwargs = dict(self.model_kwargs)
        model_load_kwargs.update(common_load_kwargs)
        if self.max_memory and "cpu" in self.max_memory:
            config = AutoConfig.from_pretrained(self.model_name, **common_load_kwargs)
            with init_empty_weights():
                skeleton = AutoModelForCausalLM.from_config(
                    config, dtype=self.torch_dtype, trust_remote_code=common_load_kwargs["trust_remote_code"]
                )
            model_load_kwargs["device_map"] = offload_device_map(skeleton, self.max_memory)
            model_load_kwargs.pop("max_memory")
            self.logger.info(f"Offload device map: {model_load_kwargs['device_map']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_load_kwargs,
        )
        load_elapsed = time.monotonic() - load_start
        self.logger.info(f"Model '{self.model_name}' loaded in {load_elapsed:.1f}s")

        if self.model_kwargs.get("device_map") != "auto":
            self.model = self.model.to(self.device)

        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        self.model.eval()

    def _offline_mode_enabled(self) -> bool:
        return _is_enabled(os.getenv("HF_HUB_OFFLINE")) or _is_enabled(os.getenv("TRANSFORMERS_OFFLINE"))

    def _resolve_hf_hub_cache_dir(self) -> Optional[str]:
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            value = os.getenv(key)
            if value:
                return value

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            return str(Path(hf_home) / "hub")
        return None

    def validate_offline_cache(self) -> None:
        if not self._offline_mode_enabled():
            return

        model_path = Path(self.model_name)
        if model_path.exists():
            return

        if "/" not in self.model_name:
            return

        cache_dir = self._resolve_hf_hub_cache_dir()
        if not cache_dir:
            raise RuntimeError(
                "Offline mode is enabled but no Hugging Face cache path is set. "
                "Set HF_HUB_CACHE (or HF_HOME) before loading model "
                f"'{self.model_name}'."
            )

        repo_dir = Path(cache_dir) / f"models--{self.model_name.replace('/', '--')}"
        if not repo_dir.exists():
            raise FileNotFoundError(
                "Offline mode is enabled and model cache is missing. "
                f"Expected repo cache directory: {repo_dir}. "
                "Prewarm the cache with huggingface_hub.snapshot_download using the same HF_HOME/HF_HUB_CACHE."
            )

    def get_input_embeddings(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embeddings = self.model.get_input_embeddings()

        new_embeddings = torch.nn.Embedding(
            num_embeddings=embeddings.num_embeddings,
            embedding_dim=embeddings.embedding_dim,
            padding_idx=embeddings.padding_idx,
            dtype=embeddings.weight.dtype,
            device=embeddings.weight.device,
        )

        with torch.no_grad():
            new_embeddings.weight.copy_(embeddings.weight)

        new_embeddings.eval()
        new_embeddings.requires_grad_(False)

        return new_embeddings

    def _get_formatted(
        self, prompts_list: Union[str, List[str]], instruction: Optional[str] = None
    ) -> List[str]:
        if isinstance(prompts_list, str):
            prompts_list = [prompts_list]

        template_kwargs: dict = {}
        fold_instruction_into_user = self.model_type in (ModelType.QWEN, ModelType.DEEPSEEK)

        tokenized_inputs = []
        for p in prompts_list:
            if fold_instruction_into_user and instruction:
                user_content = f"{instruction}\n\n{p}"
                messages = [{"role": "user", "content": user_content}]
                system_messages = []
            else:
                messages = [{"role": "user", "content": p}]
                system_messages = [{"role": "system", "content": instruction}] if instruction else []

            formatted_prompt = self.tokenizer.apply_chat_template(
                system_messages + messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            if isinstance(formatted_prompt, list):
                formatted_prompt = formatted_prompt[0] if formatted_prompt else ""

            if self.model_type == ModelType.QWEN:
                formatted_prompt = formatted_prompt + "<think>\n"

            tokenized_inputs.append(formatted_prompt)
        return tokenized_inputs

    def generate(
        self,
        prompt: Union[str, List[str]],
        instruction: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: Optional[int] = None,
        return_hidden_states: bool = False,
        return_logits: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompts_list = [prompt] if isinstance(prompt, str) else prompt

        tokenized_inputs = self._get_formatted(prompts_list, instruction=instruction)
        self.logger.info(f"Tokenized {len(tokenized_inputs)} prompts for generation")
        inputs = self.tokenizer(
            tokenized_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        first_step_hidden_states = None
        if return_hidden_states:
            self.logger.info("Extracting hidden states from forward pass")
            with torch.no_grad():
                forward_outputs = self.model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                num_layers = len(forward_outputs.hidden_states)
                self.logger.info(
                    f"Transferring hidden states to CPU: {num_layers} layers"
                )
                first_step_hidden_states = [
                    h.cpu() for h in forward_outputs.hidden_states
                ]
                self.logger.info("Hidden states transfer complete")
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        gen_kwargs = {
            "output_hidden_states": False,
            "output_scores": return_logits,
            "return_dict_in_generate": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        gen_kwargs["use_cache"] = True
        self.logger.info(f"Running generation with args: {gen_kwargs}")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        self.logger.info("Generation complete")
        generated_ids = outputs.sequences
        input_length = inputs["input_ids"].shape[1]
        generated_only_ids = generated_ids[:, input_length:]
        generated_text = self.tokenizer.batch_decode(
            generated_only_ids,
            skip_special_tokens=True,
        )
        if self.model_type == ModelType.QWEN:
            generated_text = [strip_thinking_tokens(t) for t in generated_text]
        self.logger.info("Decoded generated text")
        result = {
            "input_ids": inputs["input_ids"].cpu(),
            "generated_text": generated_text,
            "generated_ids": generated_only_ids.cpu(),
        }

        if first_step_hidden_states is not None:
            result["hidden_states"] = [first_step_hidden_states]

        if return_logits and hasattr(outputs, "scores"):
            result["logits"] = [score.cpu() for score in outputs.scores]
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        return result

    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")

        return self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

    def get_soft_token(
        self,
        text: str | List[str],
        instruction: str,
        steps: int = 1,
        add_noise: bool = True,
        gumbel_temperature: float = 0.5,
        top_k: Optional[int] = 30,
        top_p: Optional[float] = 0.95,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        formatted = (
            self._get_formatted([text], instruction)
            if isinstance(text, str)
            else self._get_formatted(text, instruction)
        )
        tokenized = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        embedding_layer = self.model.get_input_embeddings()

        current_inputs_embeds = embedding_layer(tokenized.input_ids)
        current_attention_mask = tokenized.attention_mask
        past_key_values = None

        soft_tokens_list: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(steps):
                outputs = self.model(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

                probs = torch.softmax(next_token_logits, dim=-1)

                if top_k is not None and top_k > 0:
                    probs = self._top_k_renorm_prob(probs, top_k)
                if top_p is not None and 0.0 < top_p < 1.0:
                    probs = self._top_p_renorm_prob(probs, top_p)

                actual_k = min(top_k or probs.size(-1), probs.size(-1))
                topk_probs, topk_indices = torch.topk(probs, k=actual_k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

                if add_noise:
                    topk_logits = torch.log(topk_probs.clamp(min=1e-12))
                    gumbel_noise = -torch.log(
                        -torch.log(torch.rand_like(topk_logits).clamp(min=1e-12))
                    )
                    topk_probs = torch.softmax(
                        (topk_logits + gumbel_noise) / gumbel_temperature, dim=-1
                    )
                    topk_probs, sorted_idx = torch.sort(
                        topk_probs, dim=-1, descending=True
                    )
                    topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

                topk_indices_device = topk_indices.to(embedding_layer.weight.device)
                topk_embeddings = embedding_layer.weight[topk_indices_device]
                soft_token = torch.sum(
                    topk_probs.to(topk_embeddings.device).unsqueeze(-1)
                    * topk_embeddings,
                    dim=1,
                )
                soft_tokens_list.append(soft_token)

                current_inputs_embeds = soft_token.unsqueeze(1)

                batch_size = current_inputs_embeds.shape[0]
                past_length = past_key_values[0][0].shape[2]
                current_attention_mask = torch.ones(
                    (batch_size, past_length + 1), dtype=torch.long, device=self.device
                )

        return torch.stack(soft_tokens_list, dim=1)

    def _compute_soft_token_from_logits(
        self,
        logits: torch.Tensor,
        embedding_layer,
        top_k: Optional[int],
        top_p: Optional[float],
        add_noise: bool,
        gumbel_temperature: float,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)

        if top_k is not None and top_k > 0:
            probs = self._top_k_renorm_prob(probs, top_k)
        if top_p is not None and 0.0 < top_p < 1.0:
            probs = self._top_p_renorm_prob(probs, top_p)

        actual_k = min(top_k or probs.size(-1), probs.size(-1))
        topk_probs, topk_indices = torch.topk(probs, k=actual_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        if add_noise:
            topk_logits = torch.log(topk_probs.clamp(min=1e-12))
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(topk_logits).clamp(min=1e-12))
            )
            topk_probs = torch.softmax(
                (topk_logits + gumbel_noise) / gumbel_temperature, dim=-1
            )
            topk_probs, sorted_idx = torch.sort(
                topk_probs, dim=-1, descending=True
            )
            topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)

        topk_indices_device = topk_indices.to(embedding_layer.weight.device)
        topk_embeddings = embedding_layer.weight[topk_indices_device]
        soft_token = torch.sum(
            topk_probs.to(topk_embeddings.device).unsqueeze(-1)
            * topk_embeddings,
            dim=1,
        )
        return soft_token

    @staticmethod
    def _expand_past_key_values(past_key_values, target_batch_size: int):
        from transformers.cache_utils import DynamicCache, DynamicLayer

        if isinstance(past_key_values, DynamicCache):
            expanded = DynamicCache()
            for layer in past_key_values.layers:
                new_layer = DynamicLayer()
                new_layer.dtype = layer.keys.dtype
                new_layer.device = layer.keys.device
                new_layer.is_initialized = True
                new_layer.keys = (
                    layer.keys
                    .expand(target_batch_size, -1, -1, -1)
                    .contiguous()
                )
                new_layer.values = (
                    layer.values
                    .expand(target_batch_size, -1, -1, -1)
                    .contiguous()
                )
                expanded.layers.append(new_layer)
            return expanded

        return tuple(
            (
                layer[0].expand(target_batch_size, -1, -1, -1).contiguous(),
                layer[1].expand(target_batch_size, -1, -1, -1).contiguous(),
            )
            for layer in past_key_values
        )

    def get_combined_thinking(
        self,
        text: str,
        instruction: str,
        steps: int = 128,
        gumbel_temperature: float = 0.5,
        top_k: Optional[int] = 30,
        top_p: Optional[float] = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        formatted = self._get_formatted([text], instruction)
        tokenized = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(tokenized.input_ids)
        attention_mask = tokenized.attention_mask

        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            prefill_logits = outputs.logits[:, -1, :]
            prefill_hidden = outputs.hidden_states[-1][:, -1, :]
            prefill_past = outputs.past_key_values

            soft_0 = self._compute_soft_token_from_logits(
                prefill_logits, embedding_layer, top_k, top_p,
                add_noise=False, gumbel_temperature=gumbel_temperature,
            )
            noise_0 = self._compute_soft_token_from_logits(
                prefill_logits, embedding_layer, top_k, top_p,
                add_noise=True, gumbel_temperature=gumbel_temperature,
            )
            latent_0 = self._apply_latent_realignment(prefill_hidden)

            embed_device = embedding_layer.weight.device
            soft_tokens = [soft_0]
            noise_tokens = [noise_0]
            latent_tokens = [latent_0.detach().clone().to(embed_device)]

            if steps == 1:
                return (
                    torch.stack(soft_tokens, dim=1),
                    torch.stack(noise_tokens, dim=1),
                    torch.stack(latent_tokens, dim=1),
                )

            batched_past = self._expand_past_key_values(prefill_past, 3)

            for _ in range(steps - 1):
                batched_embeds = torch.cat([
                    soft_tokens[-1].unsqueeze(1),
                    noise_tokens[-1].unsqueeze(1),
                    latent_tokens[-1].unsqueeze(1),
                ], dim=0)

                past_len = batched_past[0][0].shape[2]
                batched_mask = torch.ones(
                    (3, past_len + 1), dtype=torch.long, device=self.device
                )

                outputs = self.model(
                    inputs_embeds=batched_embeds,
                    attention_mask=batched_mask,
                    past_key_values=batched_past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
                batched_past = outputs.past_key_values
                batched_logits = outputs.logits[:, -1, :]
                batched_hidden = outputs.hidden_states[-1][:, -1, :]

                soft_tokens.append(self._compute_soft_token_from_logits(
                    batched_logits[0:1], embedding_layer, top_k, top_p,
                    add_noise=False, gumbel_temperature=gumbel_temperature,
                ))
                noise_tokens.append(self._compute_soft_token_from_logits(
                    batched_logits[1:2], embedding_layer, top_k, top_p,
                    add_noise=True, gumbel_temperature=gumbel_temperature,
                ))
                latent_vec = self._apply_latent_realignment(batched_hidden[2:3])
                latent_tokens.append(latent_vec.detach().clone().to(embed_device))

        return (
            torch.stack(soft_tokens, dim=1),
            torch.stack(noise_tokens, dim=1),
            torch.stack(latent_tokens, dim=1),
        )

    def _top_k_renorm_prob(
        self,
        probs: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        if top_k <= 0:
            return probs

        k = min(top_k, probs.size(-1))

        top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)

        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_probs)

        probs_sum = probs_filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs_filtered = probs_filtered / probs_sum

        return probs_filtered

    def _top_p_renorm_prob(
        self,
        probs: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        if not (0.0 < top_p < 1.0):
            return probs

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_probs[sorted_indices_to_remove] = 0.0

        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

        probs_sum = probs_filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs_filtered = probs_filtered / probs_sum

        return probs_filtered

    def _build_latent_realign_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = (
            self.model.get_input_embeddings()
            if hasattr(self.model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            self.model.get_output_embeddings()
            if hasattr(self.model, "get_output_embeddings")
            else None
        )
        if output_embeds is None:
            output_embeds = getattr(self.model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )
        input_weight = input_embeds.weight.detach().to(
            device=self.device, dtype=torch.float32
        )
        output_weight = output_embeds.weight.detach().to(
            device=self.device, dtype=torch.float32
        )
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        info = self._latent_realign_matrices.get(0, None)
        target_device = torch.device(self.device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix()
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )
        self._latent_realign_matrices[0] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix()
        matrix = matrix.to(hidden.device)
        target_norm = target_norm.to(hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def get_latent_thinking(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
        steps: int = 1,
    ):
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        formatted = (
            self._get_formatted([text], instruction)
            if isinstance(text, str)
            else self._get_formatted(text, instruction)
        )
        tokenized = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            embedding_layer = self.model.get_input_embeddings()
            inputs_embeds = embedding_layer(tokenized.input_ids)
            attention_mask = tokenized.attention_mask
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            latent_vecs_all: List[torch.Tensor] = []
            latent_vec = self._apply_latent_realignment(last_hidden)
            latent_vecs_all.append(latent_vec.detach().clone())
            for _ in range(steps - 1):
                latent_embed = latent_vec.unsqueeze(1)

                past_len = past[0][0].shape[2]
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                latent_vec = self._apply_latent_realignment(last_hidden)
                latent_vecs_all.append(latent_vec.detach().clone())

        return torch.stack(latent_vecs_all, dim=1)

def len_output_sequence(
    generated_ids: torch.Tensor,
    eos_token_id: int,
    num_return_sequences: int = 1,
) -> List[int]:
    if num_return_sequences is None:
        num_return_sequences = 1

    lengths = []

    for i in range(generated_ids.shape[0]):
        generated_only = generated_ids[i]

        eos_mask = generated_only == eos_token_id
        eos_positions = eos_mask.nonzero(as_tuple=True)[0]

        length = (
            min(eos_positions[0].item() + 1, len(generated_only))
            if len(eos_positions) > 0
            else len(generated_only)
        )
        lengths.append(length)

    return lengths
