"""
Configuration settings for the Modal LLM backend.
"""

from pydantic import BaseModel
from typing import Dict

MAX_CONTAINERS = 100
CONTAINER_IDLE_TIMEOUT = 60 * 5  # Scale to zero after 5 minutes idle
MAX_MODEL_LEN = 4096
LOG_LEVEL = "DEBUG"
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

class ModelConfig(BaseModel):
    """Configuration for a single model."""

    model_id: str
    gpu_type: str  # Options: T4, A10G, A100-40GB, A100-80GB, H100
    max_containers: int = MAX_CONTAINERS
    max_model_len: int = MAX_MODEL_LEN
    container_idle_timeout: int = (
        CONTAINER_IDLE_TIMEOUT  # Scale to zero after N seconds idle
    )
    quantization: str | None = None  # Options: "awq", "gptq", "fp8", None
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use (0.0-1.0)
    enforce_eager: bool = False  # Disable CUDA graphs to save memory


class ModelRegistry(BaseModel):
    """Registry of models."""

    models: Dict[str, ModelConfig]


# Available models configuration
# Only includes verified, well-supported models that exist on HuggingFace
MODEL_REGISTRY: ModelRegistry = ModelRegistry(
    models={
        "openai/gpt-oss-20b": ModelConfig(
            model_id="openai/gpt-oss-20b",
            gpu_type="A100-40GB",  # Medium-large model needs A100
            max_model_len=8192,
        ),        
        # OpenAI Open Models
        "openai/gpt-oss-120b": ModelConfig(
            model_id="openai/gpt-oss-120b",
            gpu_type="A100-80GB",  # Very large model needs A100-80GB
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        ),
        # # Mistral Models
        # "mistralai/Mistral-7B-Instruct-v0.3": ModelConfig(
        #     model_id="mistralai/Mistral-7B-Instruct-v0.3",
        #     gpu_type="A10G",  # Well-supported model, works on A10G
        #     max_model_len=4096,
        # ),
        "mistralai/Mistral-Nemo-Base-2407": ModelConfig(
            model_id="mistralai/Mistral-Nemo-Base-2407",
            gpu_type="A100-40GB",  # Large model needs A100, or use quantization
            max_model_len=4096,
            gpu_memory_utilization=0.85,  # Reduce memory usage to avoid OOM
            enforce_eager=True,  # Disable CUDA graphs to save memory
        ),
        # Google Gemma Models
        "google/gemma-2-2b-it": ModelConfig(
            model_id="google/gemma-2-2b-it",
            gpu_type="T4",  # Small model can use T4
            max_model_len=8192,
        ),
        "google/gemma-2-9b-it": ModelConfig(
            model_id="google/gemma-2-9b-it",
            gpu_type="A10G",  # Medium model can use A10G
            max_model_len=8192,
        ),
        "google/gemma-2-27b-it": ModelConfig(
            model_id="google/gemma-2-27b-it",
            gpu_type="A100-40GB",  # Large model needs A100
            max_model_len=8192,
        ),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ModelConfig(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            gpu_type="A100-40GB",  # Large model needs A100
            max_model_len=8192,
        ),
        # # Qwen Models
        # "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        #     model_id="Qwen/Qwen2.5-7B-Instruct",
        #     gpu_type="A10G",  # Medium model can use A10G
        #     max_model_len=32768,  # Qwen2.5 supports long context
        # ),
        # "Qwen/Qwen2.5-14B-Instruct": ModelConfig(
        #     model_id="Qwen/Qwen2.5-14B-Instruct",
        #     gpu_type="A100-40GB",  # Large model needs A100
        #     max_model_len=32768,
        # ),
        # "Qwen/Qwen2.5-32B-Instruct": ModelConfig(
        #     model_id="Qwen/Qwen2.5-32B-Instruct",
        #     gpu_type="A100-80GB",  # Very large model needs A100-80GB
        #     max_model_len=32768,
        # ),
    },
)

# API Configuration
APP_NAME = "llm-backend"

# For private/gated models, add your HuggingFace token
# modal secret create huggingface HF_TOKEN=hf_xxxxx
