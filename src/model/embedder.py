from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.model.base import BaseModel
from src.utils.config import EmbedderConfig
from src.utils.logging import Logger

class Embedder(BaseModel):
    def __init__(
        self,
        logger: Logger,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = "auto",
        trust_remote_code: bool = False,
        max_memory: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            logger=logger,
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            seed=seed,
            **kwargs,
        )

    @staticmethod
    def from_config(config: EmbedderConfig, logger: Logger) -> "Embedder":
        return Embedder(
            logger=logger,
            model_name=config.name,
            device=config.device,
            torch_dtype=config.torch_dtype,
            trust_remote_code=config.trust_remote_code,
            max_memory=config.max_memory,
        )

    def load(self) -> None:
        self.model = AutoModel.from_pretrained(
            self.model_name,
            **self.model_kwargs,
        ).eval()

        if self.model_kwargs.get("device_map") != "auto":
            if self.device:
                self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.model_kwargs.get("trust_remote_code", False),
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        hidden_states_masked = hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embeddings = (
            hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        )
        return F.normalize(embeddings, dim=-1)

    def embed(
        self,
        text: Union[str, List[str]],
        max_length: int = 4096,
        **kwargs,
    ) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(**inputs, **kwargs)

        return self._pool_embeddings(outputs.last_hidden_state, attention_mask)
