import enum
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

class ModelClassEnum(enum.Enum):
    BASE_LLM = "base_llm"
    EMBEDDER = "embedder"

class ModelType(enum.Enum):
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    GPT_OSS = "gpt_oss"

class ThoughtRepresentation(enum.Enum):
    num_features: Optional[int]
    is_embedding_based: bool

    def __new__(cls, value: str, num_features: Optional[int] = None, is_embedding_based: bool = False):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.num_features = num_features
        obj.is_embedding_based = is_embedding_based
        return obj

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            member = cls._value2member_map_.get(value)
            if member is not None:
                return member
        return super()._missing_(value)

    LAST_INPUT_TOKEN        = ("last_input_token",        33, False)
    LAST_INPUT_HIDDEN_STATE = ("last_input_hidden_state",  1, False)
    EMBEDDING_POOLING       = ("embedding_pooling",        1, True)
    EMBEDDING_NO_POOLING    = ("embedding_no_pooling",     1, True)
    EMBEDDING_ALL           = ("embedding_all",            1, True)
    SOFT_THINKING           = ("soft_thinking",            1, False)
    SOFT_THINKING_NOISE     = ("soft_thinking_noise",      1, False)
    LATENT_THINKING         = ("latent_thinking",          1, False)
    INPUT_EMBEDDING         = ("input_embedding",          1, True)
    RANDOM_VECTOR           = ("random_vector",            1, False)

@dataclass
class ModelConfig:
    name: str = MISSING
    model_class: ModelClassEnum = MISSING
    model_type: ModelType = MISSING
    device: str = "cuda"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    max_memory: Optional[dict] = None

@dataclass
class BaseLLMConfig(ModelConfig):
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_class: ModelClassEnum = ModelClassEnum.BASE_LLM
    model_type: ModelType = ModelType.LLAMA
    hidden_size: int = 4096
    num_hidden_layers: int = 32

    max_new_tokens: int = 16
    temperature: Optional[float] = 0
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    no_repeat_ngram_size: Optional[int] = 4
    do_sample: bool = True
    num_return_sequences: Optional[int] = None
    return_hidden_states: bool = True
    return_logits: bool = True

@dataclass
class EmbedderConfig(ModelConfig):
    name: str = "nvidia/llama-embed-nemotron-8b"
    model_class: ModelClassEnum = ModelClassEnum.EMBEDDER
    model_type: ModelType = ModelType.LLAMA
    output_dim: int = 4096

@dataclass
class LoaderConfig:
    pass

@dataclass
class BBEHLoaderConfig(LoaderConfig):
    data_dir: str = "./data/bbeh"
    task_name: Optional[str] = None
    num_examples: Optional[int] = 10

    example_start_idx: Optional[int] = None
    example_end_idx: Optional[int] = None

@dataclass
class LoggingConfig:
    name: str = "hidden_thoughts"
    log_level: int = logging.INFO
    log_dir: Optional[str] = None
    log_to_file: bool = False
    log_to_console: bool = True
    use_rich: bool = False
    log_format: Optional[str] = None

@dataclass
class ExperimentConfig:
    experiment_name: str = "dataset_curation"
    run_name: Optional[str] = None

    seed: int = 42

@dataclass
class DataLLMConfig(ExperimentConfig):
    base_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    loader: LoaderConfig = field(default_factory=BBEHLoaderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    output_dir: str = "./data"
    max_input_length: int = 4096

    save_format: str = "jsonl"
    shard_size: int = 1_024

    batch_size: int = 8

@dataclass
class DataDiscIndexConfig(ExperimentConfig):
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    output_dir: str = "./outputs/disc_index_output"
    num_return_sequences: int = 8
    total_rows: int = 480
    llm_data_output_dir: str = "./outputs/llm_data_8B/"

    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

@dataclass
class DiscDataConfig:
    base_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    num_return_sequences: int = 8
    shard_size: int = 1024
    think_steps: int = 128
    llm_data_output_dir: str = "./outputs/llm_data_output/"
    disc_index_output_dir: str = "./outputs/disc_index_output/"
    output_dir: str = "./outputs/disc_data_8B/"
    seed: int = 42
    tr_types: Optional[List[str]] = (
        None
    )

NATIVE_FAMILIES: List[str] = ["soft_thinking", "soft_thinking_noise", "latent_thinking"]
NATIVE_STEPS: List[int] = [1, 16, 32, 64, 128]

@dataclass
class CausalityNativeEvalConfig(ExperimentConfig):
    base_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    tr_data_dir: str = "./outputs/disc_data_8B/"
    llm_data_dir: str = "./outputs/llm_data_8B/"
    model_label: Optional[str] = None

    families: List[str] = field(default_factory=lambda: list(NATIVE_FAMILIES))
    steps: List[int] = field(default_factory=lambda: list(NATIVE_STEPS))
    windows: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])
    beams: int = -1
    num_examples: int = -1
    swap: bool = False
    random_vector: bool = True
    max_row_tokens: int = 4096
    bootstrap_resamples: int = 1000

    output_dir: str = "./outputs/causality_native"

@dataclass
class MinimalityNativeEvalConfig(ExperimentConfig):
    base_llm: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    tr_data_dir: str = "./outputs/disc_data_8B/"
    llm_data_dir: str = "./outputs/llm_data_8B/"
    model_label: Optional[str] = None

    families: List[str] = field(default_factory=lambda: list(NATIVE_FAMILIES))
    steps: List[int] = field(default_factory=lambda: list(NATIVE_STEPS))
    beams: int = -1
    num_examples: int = -1
    random_vector: bool = True
    max_row_tokens: int = 4096
    bootstrap_resamples: int = 1000

    output_dir: str = "./outputs/minimality_native"

def register_configs() -> None:
    pass
