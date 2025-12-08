"""
Modal GPU Backend for LLM Inference
Scales to zero when not in use, wakes up on request.

Deploy: modal deploy -m src.app
"""

import json
import modal  # type: ignore
from modal import asgi_app  # type: ignore
import logging
import traceback

from .config import APP_NAME, MAX_CONTAINERS, LOG_LEVEL, MODEL_REGISTRY as MODEL_CONFIG
from .images import api_image, vllm_image
from .api import create_app

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Persistent Volumes for Caching
# ---------------------------------------------------------------------------

# Cache volumes for faster model loading and reduced cold start times
# These volumes persist model weights and vLLM compilation artifacts across container restarts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Model Classes (must be defined at global scope for Modal)
# ---------------------------------------------------------------------------

class BaseModel:
    """Base class for vLLM model servers with shared functionality."""
    
    # These will be overridden by subclasses
    _model_id: str
    _gpu_type: str
    _max_model_len: int
    _quantization: str | None
    _gpu_memory_utilization: float
    _enforce_eager: bool
    
    @modal.enter()
    def load_model(self):
        """Load model once when container starts (cached across requests)."""
        import os
        from vllm import LLM, SamplingParams  # type: ignore
        
        print(f"Loading model: {self._model_id}")
        print(f"Config: GPU={self._gpu_type}, MaxLen={self._max_model_len}, Quantization={self._quantization}, GPUUtil={self._gpu_memory_utilization}")
        
        # Check for HuggingFace token (required for gated models)
        # Modal secrets automatically inject environment variables
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            # Ensure both environment variable names are set (transformers uses either)
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print("✅ HuggingFace token found - can access gated models")
        else:
            print("⚠️  Warning: No HuggingFace token found. Gated models may fail.")
            print("   To fix: modal secret create huggingface HF_TOKEN=hf_xxxxx")
            print("   Also ensure you've accepted the model's terms at:")
            print(f"   https://huggingface.co/{self._model_id}")
        
        # Set cache directories from mounted volumes
        # HuggingFace cache for model weights
        hf_cache_dir = "/cache/huggingface"
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        
        # vLLM cache for compilation artifacts
        vllm_cache_dir = "/cache/vllm"
        os.makedirs(vllm_cache_dir, exist_ok=True)
        os.environ["VLLM_CACHE_DIR"] = vllm_cache_dir
        
        print(f"Using cache directories: HF={hf_cache_dir}, vLLM={vllm_cache_dir}")
        
        # Build LLM kwargs
        # vLLM will automatically use the HuggingFace cache directories set via environment variables
        llm_kwargs = {
            "model": self._model_id,
            "max_model_len": self._max_model_len,
            "trust_remote_code": True,
            "gpu_memory_utilization": self._gpu_memory_utilization,
        }
        
        # Add quantization if specified
        if self._quantization:
            llm_kwargs["quantization"] = self._quantization
        
        # Add enforce_eager if specified (saves memory)
        if self._enforce_eager:
            llm_kwargs["enforce_eager"] = True
        
        self.llm = LLM(**llm_kwargs)
        self.default_sampling = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )
        
        # Commit volumes to persist cache after model loading
        # This ensures model weights and compilation artifacts are saved for faster subsequent loads
        hf_cache_vol.commit()
        vllm_cache_vol.commit()
        
        print("Model loaded and cache committed!")
    
    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Generate completion for a prompt."""
        from vllm import SamplingParams
        
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate([prompt], sampling)
        return outputs[0].outputs[0].text
    
    @modal.method()
    def chat(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
        """Chat completion (OpenAI format)."""
        from vllm import SamplingParams
        
        # Format messages for the model
        prompt = self._format_messages(messages)
        
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate([prompt], sampling)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": outputs[0].outputs[0].text
                }
            }]
        }
    
    @modal.method()
    def chat_stream(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
        """Stream chat completion (OpenAI format). Returns list of text deltas."""
        from vllm import SamplingParams
        
        try:
            # Validate input
            if not messages or not isinstance(messages, list):
                raise ValueError("messages must be a non-empty list")
            
            # Format messages for the model
            prompt = self._format_messages(messages)
            logger.debug(f"chat_stream: Formatted prompt length: {len(prompt)}")
            
            if not prompt or not prompt.strip():
                raise ValueError("Formatted prompt is empty")
            
            sampling = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            logger.debug("chat_stream: Starting vLLM generation")
            
            request_outputs = self.llm.generate([prompt], sampling, use_tqdm=False)
            
            if not request_outputs:
                raise ValueError("vLLM returned empty outputs")
            
            request_output = request_outputs[0]
            if not hasattr(request_output, 'outputs') or not request_output.outputs:
                raise ValueError("RequestOutput has no outputs")
            
            # Get the final generated text
            final_text = request_output.outputs[0].text
            
            # Simulate streaming by chunking the text into smaller pieces
            chunk_size = 10  # Characters per chunk
            chunks = []
            for i in range(0, len(final_text), chunk_size):
                chunk = final_text[i:i + chunk_size]
                if chunk:
                    chunks.append(chunk)
            
            # If text is very short, return it as a single chunk
            if not chunks and final_text:
                chunks = [final_text]
            
            logger.debug(f"chat_stream: Generated {len(chunks)} chunks from text of length {len(final_text)}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"chat_stream: Error in chat_stream method: {str(e)}")
            logger.error(f"chat_stream: Traceback: {traceback.format_exc()}")
            raise
    
    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages for Mistral-style models."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"[INST] {content} [/INST]\n"
            elif role == "user":
                formatted += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                formatted += f"{content}\n"
        return formatted


# ---------------------------------------------------------------------------
# Model Registration
# ---------------------------------------------------------------------------

def register_model(model_id: str, model_config) -> type:
    """
    Register a model with Modal by creating a configured Modal class.
    
    Each model gets its own Modal class with its own container that scales independently.
    The class is created at module level so Modal can properly register @modal.method() decorators.
    
    Args:
        model_id: The model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        model_config: ModelConfig instance with model-specific settings
        
    Returns:
        A Modal class decorated with app.cls() and configured for the specific model
    """
    # Create a clean class name from model_id
    class_name = f"Model_{model_id.replace('/', '_').replace('-', '_').replace('.', '_')}"
    
    # Create a subclass of BaseModel with model-specific configuration
    ModelClass = type(
        class_name,
        (BaseModel,),
        {
            "_model_id": model_id,
            "_gpu_type": model_config.gpu_type,
            "_max_model_len": model_config.max_model_len,
            "_quantization": model_config.quantization,
            "_gpu_memory_utilization": model_config.gpu_memory_utilization,
            "_enforce_eager": model_config.enforce_eager,
            "__doc__": f"vLLM model server for {model_id}.",
        }
    )
    
    # Apply Modal decorator with model-specific settings
    # This creates a Modal class that will run in its own container
    # Mount volumes for caching model weights and compilation artifacts
    ModalModel = app.cls(
        gpu=model_config.gpu_type,
        image=vllm_image,
        scaledown_window=model_config.container_idle_timeout,
        secrets=[modal.Secret.from_name("huggingface")],  # For gated/private models on HuggingFace
        max_containers=model_config.max_containers,
        volumes={
            "/cache/huggingface": hf_cache_vol,
            "/cache/vllm": vllm_cache_vol,
        },
    )(ModelClass)
    
    # Add to module namespace so Modal can discover the class
    # This is required for Modal to properly register @modal.method() decorators
    # at module import time
    globals()[class_name] = ModalModel
    
    return ModalModel


# Register all models from configuration
# Each model gets its own container that scales independently
MODEL_REGISTRY: dict[str, type] = {
    model_id: register_model(model_id, model_config)
    for model_id, model_config in MODEL_CONFIG.models.items()
}

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
    # Log available model IDs (MODEL_REGISTRY contains Modal classes which aren't JSON serializable)
    available_models = list(MODEL_REGISTRY.keys())
    logger.debug(f"Available models: {json.dumps(available_models, indent=2)}")
    return create_app(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Test the model locally."""
    # Use default model for testing
    default_model_id = list(MODEL_REGISTRY.keys())[0]
    ModelClass = MODEL_REGISTRY[default_model_id]
    model = ModelClass()

    # Test chat
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    result = model.chat.remote(messages)
    print("Response:", result["choices"][0]["message"]["content"])

