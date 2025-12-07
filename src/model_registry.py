import modal
import logging
from src.config import MODEL_REGISTRY
from src.model import create_model_class
from src.images import vllm_image

logger = logging.getLogger(__name__)

def get_model_registry(app: modal.App) -> dict:
    """Get the model registry."""
    available_models = MODEL_REGISTRY.models

    logger.info(f"Model registry: {available_models}")

    # Create a Modal class for each model
    # Each model gets its own container that scales independently
    model_registry = {}

    for model_id, model_config in available_models.items():
        # Create a Model class for this specific model
        ModelClass = create_model_class(model_id)
        
        # Apply Modal decorator with model-specific settings
        ModalModel = app.cls(
            gpu=model_config.gpu_type,
            image=vllm_image,
            scaledown_window=model_config.container_idle_timeout,
            # secrets=[modal.Secret.from_name("huggingface")],  # Optional: only needed for private/gated models
            max_containers=model_config.max_containers,
        )(ModelClass)
        
        # Store in registry for routing
        model_registry[model_id] = ModalModel

    return model_registry