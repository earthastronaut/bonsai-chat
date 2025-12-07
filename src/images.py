"""
Container image definitions for Modal.
"""

import modal
from .config import MODEL_ID

# vLLM image for model inference
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "transformers",
        "torch",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# FastAPI image for API endpoints
api_image = modal.Image.debian_slim().pip_install("fastapi", "pydantic")

