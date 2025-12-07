"""
Configuration settings for the Modal LLM backend.
"""

from pydantic import BaseModel
from typing import Dict

MAX_CONTAINERS = 100
CONTAINER_IDLE_TIMEOUT = 60 * 5  # Scale to zero after 5 minutes idle
MAX_MODEL_LEN = 4096
LOG_LEVEL = "DEBUG"
DEFAULT_MODEL_ID = "openai/gpt-oss-120b"

class ModelConfig(BaseModel):
    """Configuration for a single model."""

    model_id: str
    gpu_type: str  # Options: T4, A10G, A100-40GB, A100-80GB, H100
    max_containers: int = MAX_CONTAINERS
    max_model_len: int = MAX_MODEL_LEN
    container_idle_timeout: int = (
        CONTAINER_IDLE_TIMEOUT  # Scale to zero after N seconds idle
    )


class ModelRegistry(BaseModel):
    """Registry of models."""

    models: Dict[str, ModelConfig]


# Available models configuration
MODEL_REGISTRY: ModelRegistry = ModelRegistry(
    models={
        "openai/gpt-oss-120b": ModelConfig(
            model_id="openai/gpt-oss-120b",
            gpu_type="A100-80GB",  # Large model needs A100
            max_model_len=8192,
        ),
        "Qwen/Qwen3-235B-A22B": ModelConfig(
            model_id="Qwen/Qwen3-235B-A22B",
            gpu_type="A100-80GB",  # Very large model needs A100
            max_model_len=8192,
        ),
        "Qwen/Qwen3-30B-A3B-T": ModelConfig(
            model_id="Qwen/Qwen3-30B-A3B-T",
            gpu_type="A10G",  # Medium model can use A10G
            max_model_len=4096,
        ),
        "mistralai/Mistral-7B-Instruct-v0.3": ModelConfig(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            gpu_type="A10G",  # Small model can use A10G
            max_model_len=4096,
        ),
        "mistralai/Mistral-Nemo-Base-2407": ModelConfig(
            model_id="mistralai/Mistral-Nemo-Base-2407",
            gpu_type="A10G",  # Small model can use A10G
            max_model_len=4096,
        ),
    },
)

# Legacy GPU config (for backward compatibility)
GPU_TYPE = "A10G"

# API Configuration
APP_NAME = "llm-backend"

# For private/gated models, add your HuggingFace token
# modal secret create huggingface HF_TOKEN=hf_xxxxx
