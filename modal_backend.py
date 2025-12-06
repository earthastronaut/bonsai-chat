"""
Modal GPU Backend for LLM Inference
Scales to zero when not in use, wakes up on request.

Deploy: modal deploy modal_backend.py
"""

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Change to your preferred model
GPU_TYPE = "A10G"  # Options: T4, A10G, A100-40GB, A100-80GB, H100
MAX_MODEL_LEN = 4096
CONTAINER_IDLE_TIMEOUT = 120  # Scale to zero after 2 min idle

# For private/gated models, add your HuggingFace token
# modal secret create huggingface HF_TOKEN=hf_xxxxx

# ---------------------------------------------------------------------------
# Container Image
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "transformers",
        "torch",
        "huggingface_hub",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = modal.App("llm-backend")

@app.cls(
    gpu=GPU_TYPE,
    image=vllm_image,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=100,
    secrets=[modal.Secret.from_name("huggingface", required_hint=False)],
)
class Model:
    """vLLM model server that scales to zero."""

    @modal.enter()
    def load_model(self):
        """Load model once when container starts (cached across requests)."""
        from vllm import LLM, SamplingParams

        print(f"Loading model: {MODEL_ID}")
        self.llm = LLM(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )
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
# OpenAI-Compatible API (for Open WebUI)
# ---------------------------------------------------------------------------

from modal import asgi_app
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import uuid

web_app = FastAPI(title="LLM API")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API key auth - set via: modal secret create api-auth API_KEY=your-secret-key
API_KEY = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7

@web_app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

@web_app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, authorization: str = Header(None)):
    """Chat completions endpoint (OpenAI compatible)."""
    # Auth check
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if request.stream:
        # Streaming not implemented in this example
        raise HTTPException(status_code=400, detail="Streaming not supported yet")
    
    model = Model()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    result = model.chat.remote(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": result["choices"],
        "usage": {
            "prompt_tokens": -1,  # vLLM can provide this
            "completion_tokens": -1,
            "total_tokens": -1,
        }
    }

@web_app.post("/v1/completions")
async def completions(request: CompletionRequest, authorization: str = Header(None)):
    """Text completions endpoint (OpenAI compatible)."""
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    model = Model()
    text = model.generate.remote(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }]
    }

@web_app.get("/health")
async def health():
    return {"status": "ok"}


@app.function(
    image=modal.Image.debian_slim().pip_install("fastapi", "pydantic"),
    secrets=[modal.Secret.from_name("api-auth", required_hint=False)],
    allow_concurrent_inputs=100,
)
@asgi_app()
def api():
    """OpenAI-compatible API endpoint."""
    import os
    global API_KEY
    API_KEY = os.environ.get("API_KEY")
    return web_app


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

