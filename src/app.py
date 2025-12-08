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
        
        # Store tokenizer reference for chat template formatting
        # Access tokenizer through llm_engine which is available after LLM initialization
        self.tokenizer = self.llm.llm_engine.tokenizer if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'tokenizer') else None
        
        # Detect if this is a reasoning model and set up stop sequences
        # Reasoning models often output "analysis...final<answer>" pattern
        # We can use stop sequences or extract after "final" marker
        self.is_reasoning_model = False
        if self.tokenizer:
            # Check for common reasoning model patterns or special tokens
            special_tokens = getattr(self.tokenizer, 'special_tokens_map', {})
            # Some reasoning models have special tokens like "final_answer" or similar
            if any('final' in str(v).lower() for v in special_tokens.values()):
                self.is_reasoning_model = True
            # Also check model name patterns
            if 'oss' in self._model_id.lower() or 'reasoning' in self._model_id.lower():
                self.is_reasoning_model = True
        
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
        
        # For reasoning models, we might want to use stop sequences
        # But first, let's try using the tokenizer to extract only new tokens
        stop_sequences = None
        # Note: We extract the final answer after generation using pattern matching
        # Some reasoning models might benefit from stop sequences, but pattern matching
        # is more general and works across different model formats
        
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )
        
        outputs = self.llm.generate([prompt], sampling)
        
        # vLLM already returns only the generated text (not the prompt)
        # But for reasoning models, we need to extract the final answer from the reasoning
        raw_text = outputs[0].outputs[0].text
        
        # Extract only the final answer using the tokenizer's knowledge if available
        # or fall back to pattern matching
        cleaned_text = self._extract_final_answer(raw_text, prompt)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": cleaned_text
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
            
            # Get the final generated text and extract final answer
            raw_text = request_output.outputs[0].text
            final_text = self._extract_final_answer(raw_text, prompt)
            
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
    
    def _extract_final_answer(self, generated_text: str, prompt: str = "") -> str:
        """Extract only the final answer from model output.
        
        For reasoning models, this extracts content after "final" markers.
        For regular models, returns the text as-is (vLLM already excludes the prompt).
        
        Args:
            generated_text: The text generated by the model (already excludes prompt)
            prompt: The original prompt (for reference, not used in extraction)
            
        Returns:
            The final answer text, cleaned of reasoning/analysis prefixes
        """
        if not generated_text:
            return generated_text
        
        # First, try to use tokenizer-based extraction if this is a reasoning model
        if self.is_reasoning_model and self.tokenizer:
            # Check if tokenizer has special tokens for final answers
            try:
                # Some models use special tokens - check if we can decode and find them
                # For now, fall through to pattern matching
                pass
            except Exception:
                pass
        
        # Use pattern matching to extract final answer
        # This handles models that output "analysis...final<answer>" pattern
        return self._clean_output(generated_text)
    
    def _clean_output(self, text: str) -> str:
        """Remove instruction tags, analysis prefixes, and other formatting artifacts from model output.
        
        Extracts only the final answer, removing reasoning/analysis sections that some models include.
        """
        if not text:
            return text
        
        import re
        
        # Remove [INST]...[/INST] patterns
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        # Remove standalone [INST] or [/INST] tags
        text = re.sub(r'\[/?INST\]', '', text)
        
        # For models that include analysis/reasoning, extract only content after "final" marker
        # Pattern: "analysis...assistantfinal<actual answer>"
        # Look for "final" (case insensitive) and extract everything after it
        # Handle both "final<answer>" and "final <answer>" patterns
        final_match = re.search(r'final\s*(.+)$', text, re.IGNORECASE | re.DOTALL)
        if final_match:
            text = final_match.group(1)
        else:
            # Also try pattern where "assistant" and "final" are concatenated: "assistantfinal"
            assistant_final_match = re.search(r'assistantfinal\s*(.+)$', text, re.IGNORECASE | re.DOTALL)
            if assistant_final_match:
                text = assistant_final_match.group(1)
            else:
                # If no "final" marker, try "assistant" marker alone
                assistant_match = re.search(r'assistant\s+(.+)$', text, re.IGNORECASE | re.DOTALL)
                if assistant_match:
                    text = assistant_match.group(1)
                else:
                    # Remove "analysis" prefix and any reasoning text
                    # Pattern: "analysis" followed by text, then "assistant" or other markers
                    text = re.sub(r'^analysis.*?(?:assistant|final)\s*', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove any remaining role markers at the start (with or without colon)
        text = re.sub(r'^(analysis|assistant|user|system)[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Remove prompt echo - if text starts with something that looks like the user's question
        # followed by the answer, try to extract just the answer part
        # This is a heuristic: look for common question patterns followed by the actual answer
        # But be conservative - only remove if we're confident it's an echo
        
        # Clean up extra whitespace and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = text.strip()
        
        return text
    
    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages using the model's tokenizer chat template."""
        # Convert messages to the format expected by apply_chat_template
        # The tokenizer expects a list of dicts with 'role' and 'content' keys
        chat_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # Include all messages (system, user, assistant) for proper chat template formatting
            # The chat template will handle the formatting correctly
            chat_messages.append({"role": role, "content": content})
        
        # Use the tokenizer's apply_chat_template to format messages correctly
        # This ensures the model-specific chat template is used (e.g., Mistral, Llama, etc.)
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # apply_chat_template with tokenize=False returns a string
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                if isinstance(prompt, str):
                    return prompt
                else:
                    # If it returns a list (tokenized), convert back to string
                    logger.warning("Tokenizer returned tokenized output, converting to string")
                    return self.tokenizer.decode(prompt) if isinstance(prompt, list) else str(prompt)
            except Exception as e:
                # Fallback: if apply_chat_template fails, use a simple format
                logger.warning(f"Failed to apply chat template: {e}, using fallback format")
        else:
            logger.warning("Tokenizer not available, using fallback format")
        
        # Fallback: simple format without chat template
        formatted = ""
        for msg in chat_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
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

