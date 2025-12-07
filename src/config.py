"""
Configuration settings for the Modal LLM backend.
"""

# Model Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Change to your preferred model
GPU_TYPE = "A10G"  # Options: T4, A10G, A100-40GB, A100-80GB, H100
MAX_MODEL_LEN = 4096
CONTAINER_IDLE_TIMEOUT = 120  # Scale to zero after 2 min idle

# API Configuration
APP_NAME = "llm-backend"
MAX_CONTAINERS = 100

# For private/gated models, add your HuggingFace token
# modal secret create huggingface HF_TOKEN=hf_xxxxx

