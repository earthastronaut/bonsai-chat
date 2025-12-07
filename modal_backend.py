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
import traceback

from src.config import APP_NAME, MAX_CONTAINERS, LOG_LEVEL, MODEL_REGISTRY as MODEL_CONFIG
from src.images import api_image, vllm_image
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
        from vllm import LLM, SamplingParams  # type: ignore
        
        print(f"Loading model: {self._model_id}")
        print(f"Config: GPU={self._gpu_type}, MaxLen={self._max_model_len}, Quantization={self._quantization}, GPUUtil={self._gpu_memory_utilization}")
        
        # Build LLM kwargs
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
        print("Model loaded!")
    
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


# Create a Modal class for each model
# Each model gets its own container that scales independently
# NOTE: Classes must be defined at module level for Modal to see @modal.method() decorators
MODEL_REGISTRY = {}

for model_id, model_config in MODEL_CONFIG.models.items():
    # Create a unique class name for this model
    class_name = f"Model_{model_id.replace('/', '_').replace('-', '_').replace('.', '_')}"
    
    # Create a subclass of BaseModel with model-specific configuration
    # This is done at module level, so Modal can see the decorators
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
    ModalModel = app.cls(
        gpu=model_config.gpu_type,
        image=vllm_image,
        scaledown_window=model_config.container_idle_timeout,
        # secrets=[modal.Secret.from_name("huggingface")],  # Optional: only needed for private/gated models
        max_containers=model_config.max_containers,
    )(ModelClass)
    
    # Add to module namespace so Modal can import it
    globals()[class_name] = ModalModel
    
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
