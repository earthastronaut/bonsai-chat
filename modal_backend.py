"""
Modal GPU Backend for LLM Inference
Scales to zero when not in use, wakes up on request.

Deploy: modal deploy modal_backend.py

Note: The src/ directory is explicitly included in the Modal images via add_local_dir,
so all modules in src/ will be packaged and available at runtime.
"""

import modal
from modal import asgi_app

from src.config import APP_NAME, MAX_CONTAINERS, GPU_TYPE, CONTAINER_IDLE_TIMEOUT
from src.images import api_image, vllm_image
from src.model import Model
from src.api import create_app

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Apply Modal decorator to Model class
Model = app.cls(
    gpu=GPU_TYPE,
    image=vllm_image,
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    # secrets=[modal.Secret.from_name("huggingface")],  # Optional: only needed for private/gated models
    max_containers=MAX_CONTAINERS,
)(Model)

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
    return create_app(Model)


# ---------------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Test the model locally."""
    model = Model()
    
    # Test chat
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    result = model.chat.remote(messages)
    print("Response:", result["choices"][0]["message"]["content"])
