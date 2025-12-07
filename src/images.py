"""
Container image definitions for Modal.

When deploying as a package (modal deploy -m src.app), Modal automatically
includes all local modules, so we don't need add_local_dir.
"""

import modal  # type: ignore

# vLLM image for model inference
# Includes extra dependencies for headroom and compatibility
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",  # For trust_remote_code models
        "curl",  # For debugging and health checks
        "build-essential",  # For compiling some model dependencies
    )
    .pip_install(
        "vllm>=0.6.0",
        "transformers>=4.40.0",  # Ensure compatibility
        "torch>=2.1.0",
        "torchvision",
        "torchaudio",
        "huggingface_hub>=0.20.0",
        "hf_transfer>=0.1.0",  # Faster model downloads
        "accelerate>=0.25.0",  # Model loading utilities
        "sentencepiece",  # Tokenizer support
        "protobuf>=3.20.0",  # Protocol buffer support
        "ninja",  # Build tool for some dependencies
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Reduce memory fragmentation
        "CUDA_LAUNCH_BLOCKING": "0",  # Allow async CUDA operations
    })
)

# FastAPI image for API endpoints
# Includes all dependencies needed for the API server
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",  # ASGI server
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",  # For settings management
        "httpx>=0.25.0",  # HTTP client for health checks
        "python-multipart",  # For form data parsing
        "starlette>=0.27.0",  # FastAPI dependency
    )
    .env({
        "PYTHONUNBUFFERED": "1",  # Ensure logs are flushed
    })
)

