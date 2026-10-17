from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers.auto import AutoHfQuantizer

from src.utils import get_attn_impl
from src.utils.logging import Logger

def _patch_transformers_quantization_config() -> None:
    original_to_dict = PretrainedConfig.to_dict
    original_to_diff_dict = PretrainedConfig.to_diff_dict

    def _without_none_qc(self: PretrainedConfig, fn):
        qc = self.__dict__.get("quantization_config")
        if qc is None and "quantization_config" in self.__dict__:
            del self.__dict__["quantization_config"]
            try:
                return fn(self)
            finally:
                self.__dict__["quantization_config"] = None
        return fn(self)

    def _safe_to_dict(self: PretrainedConfig) -> Dict:
        return _without_none_qc(self, original_to_dict)

    def _safe_to_diff_dict(self: PretrainedConfig) -> Dict:
        return _without_none_qc(self, original_to_diff_dict)

    PretrainedConfig.to_dict = _safe_to_dict
    PretrainedConfig.to_diff_dict = _safe_to_diff_dict

    _orig_supports_quant_method = AutoHfQuantizer.supports_quant_method

    @staticmethod
    def _safe_supports_quant_method(quantization_config_dict: Any) -> bool:
        if quantization_config_dict is None:
            return False
        return _orig_supports_quant_method(quantization_config_dict)

    AutoHfQuantizer.supports_quant_method = _safe_supports_quant_method

_patch_transformers_quantization_config()

class BaseModel(ABC):
    logger: Logger
    device: str
    model_name: str
    torch_dtype: torch.dtype | str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    model_kwargs: Dict[str, Any]

    def __init__(
        self,
        logger: Logger,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = "auto",
        trust_remote_code: bool = False,
        max_memory: Optional[Dict[str, str]] = None,
        model_type: Any = None,
        **kwargs,
    ):
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.torch_dtype: torch.dtype | str = torch_dtype

        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested (device={self.device}) but torch.cuda.is_available() is False. "
                f"Check the GPU with nvidia-smi."
            )

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "dtype": self.torch_dtype,
        }

        if self.device.startswith("cuda"):
            if ":" not in self.device:
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": self.device}
        else:
            model_kwargs["low_cpu_mem_usage"] = True

        attn_impl = get_attn_impl(self.torch_dtype, model_type=model_type)
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        if max_memory:
            max_memory = {int(k) if str(k).isdigit() else str(k): v for k, v in max_memory.items()}
            model_kwargs["max_memory"] = max_memory
        self.max_memory = max_memory
        self.logger.info(
            "Resolved loading config: "
            f"device={self.device}, "
            f"device_map={model_kwargs.get('device_map')}, "
            f"attn_implementation={model_kwargs.get('attn_implementation')}"
        )
        self.logger.info(f"Loading with model kwargs: {model_kwargs}")
        self.model_kwargs = model_kwargs

    @abstractmethod
    def load(self) -> None:
        pass

    def to(self, device: str) -> "BaseModel":
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self) -> "BaseModel":
        if self.model is not None:
            self.model.eval()
        return self

    def train(self, mode: bool = True) -> "BaseModel":
        if self.model is not None:
            self.model.train(mode)
        return self
