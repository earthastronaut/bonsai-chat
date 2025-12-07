"""
Modal GPU Backend for LLM Inference
Scales to zero when not in use, wakes up on request.

Deploy: modal deploy modal_backend.py

Note: The src/ directory is explicitly included in the Modal images via add_local_dir,
so all modules in src/ will be packaged and available at runtime.
"""

import json
import modal  # type: ignore
from modal import asgi_app  # type: ignore
import logging

from src.config import APP_NAME, MAX_CONTAINERS, LOG_LEVEL, AVAILABLE_MODELS, DEFAULT_MODEL_ID
from src.images import api_image, vllm_image
from src.model import create_model_class
from src.api import create_app

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Create a Modal class for each model
# Each model gets its own container that scales independently
MODEL_REGISTRY = {}

for model_id, model_config in AVAILABLE_MODELS.items():
    # Create a Model class for this specific model
    ModelClass = create_model_class(model_id)
    
    # Apply Modal decorator with model-specific settings
    ModalModel = app.cls(
        gpu=model_config.gpu_type,
        image=vllm_image,
        scaledown_window=model_config.container_idle_timeout,
        # secrets=[modal.Secret.from_name("huggingface")],  # Optional: only needed for private/gated models
        max_containers=MAX_CONTAINERS,
    )(ModelClass)
    
    # Store in registry for routing
    MODEL_REGISTRY[model_id] = ModalModel

# ---------------------------------------------------------------------------
# OpenAI-Compatible API (for Open WebUI)
# ---------------------------------------------------------------------------

@app.function(
    image=api_image,
    secrets=[modal.Secret.from_name("api-auth")],
    max_containers=MAX_CONTAINERS,
)
@asgi_app()
def api():
    """OpenAI-compatible API endpoint."""
    logger.debug(f"Available models: {json.dumps(MODEL_REGISTRY, indent=2)}")
    return create_app(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Test the model locally."""
    # Use default model for testing
    ModelClass = MODEL_REGISTRY[DEFAULT_MODEL_ID]
    model = ModelClass()
    
    # Test chat
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    result = model.chat.remote(messages)
    print("Response:", result["choices"][0]["message"]["content"])
