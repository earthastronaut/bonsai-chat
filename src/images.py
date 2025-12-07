"""
Container image definitions for Modal.
"""

import modal
from pathlib import Path

# Get the src directory path
# When Modal packages, it resolves paths relative to the deployment root
# Since modal_backend.py is at root and imports from src, we reference src/ relative to root
SRC_DIR = Path(__file__).parent

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
    .add_local_dir("src", remote_path="/pkg/src")
    .env({"PYTHONPATH": "/pkg"})
)

# FastAPI image for API endpoints
api_image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "pydantic")
    .add_local_dir("src", remote_path="/pkg/src")
    .env({"PYTHONPATH": "/pkg"})
)

