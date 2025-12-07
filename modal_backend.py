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
    scaledown_window=CONTAINER_IDLE_TIMEOUT,
    # secrets=[modal.Secret.from_name("huggingface")],  # Optional: only needed for private/gated models
    max_containers=100,
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

    @modal.method()
    def chat_stream(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
        """Stream chat completion (OpenAI format). Returns list of text deltas."""
        from vllm import SamplingParams

        # Format messages for the model
        prompt = self._format_messages(messages)
        
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Use streaming generation - vLLM yields RequestOutput objects incrementally
        stream = self.llm.generate([prompt], sampling, use_tqdm=False)
        
        # Collect text deltas as they're generated
        chunks = []
        previous_text = ""
        for output in stream:
            # Each output contains the full text so far, extract the delta
            for choice in output.outputs:
                current_text = choice.text
                # Calculate delta (new text since last chunk)
                if len(current_text) > len(previous_text):
                    delta = current_text[len(previous_text):]
                    if delta:
                        chunks.append(delta)
                        previous_text = current_text
        
        return chunks

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
import time
import uuid


@app.function(
    image=modal.Image.debian_slim().pip_install("fastapi", "pydantic"),
    secrets=[modal.Secret.from_name("api-auth")],
    max_containers=100,
)
@asgi_app()
def api():
    """OpenAI-compatible API endpoint."""
    from fastapi import FastAPI, HTTPException, Header, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import os
    import json
    import logging
    import traceback
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("llm-api")
    
    web_app = FastAPI(title="LLM API")
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @web_app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests."""
        start_time = time.time()
        request_id = uuid.uuid4().hex[:8]
        
        # Log request details
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Log headers (sanitize sensitive ones)
        headers = dict(request.headers)
        if "authorization" in headers:
            auth_header = headers["authorization"]
            if auth_header.startswith("Bearer "):
                token_preview = auth_header[:20] + "..." if len(auth_header) > 20 else auth_header
                headers["authorization"] = token_preview
        logger.debug(f"[{request_id}] Headers: {json.dumps(headers, indent=2)}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.3f}s"
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s"
            )
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
            raise
    
    # Simple API key auth - set via: modal secret create api-auth API_KEY=your-secret-key
    API_KEY = os.environ.get("API_KEY")
    
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
        logger.info("GET /v1/models - Listing available models")
        try:
            response = {
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }]
            }
            logger.info(f"GET /v1/models - Success: {len(response['data'])} model(s) returned")
            return response
        except Exception as e:
            logger.error(f"GET /v1/models - Error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @web_app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest, authorization: str = Header(None)):
        """Chat completions endpoint (OpenAI compatible)."""
        request_start = time.time()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        # Log request details
        logger.info(
            f"POST /v1/chat/completions - Request ID: {completion_id} - "
            f"Model: {request.model}, Stream: {request.stream}, "
            f"Messages: {len(request.messages)}, Max tokens: {request.max_tokens}, "
            f"Temperature: {request.temperature}"
        )
        
        # Log message structure (sanitize content for privacy)
        message_preview = []
        for i, msg in enumerate(request.messages):
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            message_preview.append(f"msg[{i}]: role={msg.role}, content_len={len(msg.content)}, preview='{content_preview}'")
        logger.debug(f"Request {completion_id} - Messages: {'; '.join(message_preview)}")
        
        # Auth check
        auth_required = API_KEY is not None
        auth_provided = authorization is not None
        auth_valid = False
        
        if auth_required:
            if not auth_provided:
                logger.warning(f"Request {completion_id} - Authentication failed: No authorization header provided")
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            auth_valid = authorization == f"Bearer {API_KEY}"
            if not auth_valid:
                logger.warning(
                    f"Request {completion_id} - Authentication failed: Invalid API key "
                    f"(provided: {authorization[:20]}...)"
                )
                raise HTTPException(status_code=401, detail="Invalid API key")
            logger.info(f"Request {completion_id} - Authentication successful")
        else:
            logger.debug(f"Request {completion_id} - No authentication required (API_KEY not set)")
        
        try:
            model = Model()
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
            if request.stream:
                logger.info(f"Request {completion_id} - Starting streaming response")
                stream_start = time.time()
                
                async def generate_stream():
                    """Generate streaming response in SSE format."""
                    try:
                        # Send initial chunk with metadata
                        initial_chunk = {
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': request.model,
                            'choices': [{
                                'index': 0,
                                'delta': {'role': 'assistant', 'content': ''},
                                'finish_reason': None
                            }]
                        }
                        yield f"data: {json.dumps(initial_chunk)}\n\n"
                        logger.debug(f"Request {completion_id} - Sent initial chunk")
                        
                        # Get streaming chunks from model
                        model_call_start = time.time()
                        logger.info(f"Request {completion_id} - Calling model.chat_stream.remote()")
                        chunks = model.chat_stream.remote(
                            messages=messages,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                        )
                        model_call_time = time.time() - model_call_start
                        logger.info(
                            f"Request {completion_id} - Model call completed in {model_call_time:.3f}s, "
                            f"received {len(chunks)} chunks"
                        )
                        
                        # Stream each chunk
                        chunk_count = 0
                        total_content_length = 0
                        for delta_content in chunks:
                            if delta_content:
                                chunk_count += 1
                                total_content_length += len(delta_content)
                                chunk_data = {
                                    'id': completion_id,
                                    'object': 'chat.completion.chunk',
                                    'created': created,
                                    'model': request.model,
                                    'choices': [{
                                        'index': 0,
                                        'delta': {'content': delta_content},
                                        'finish_reason': None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                        
                        logger.debug(
                            f"Request {completion_id} - Streamed {chunk_count} chunks, "
                            f"total content length: {total_content_length} chars"
                        )
                        
                        # Send final chunk with finish_reason
                        final_chunk = {
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': request.model,
                            'choices': [{
                                'index': 0,
                                'delta': {},
                                'finish_reason': 'stop'
                            }]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        
                        # Send final [DONE] marker
                        yield "data: [DONE]\n\n"
                        
                        stream_time = time.time() - stream_start
                        logger.info(
                            f"Request {completion_id} - Streaming completed in {stream_time:.3f}s, "
                            f"total chunks: {chunk_count}, total length: {total_content_length} chars"
                        )
                    except Exception as e:
                        logger.error(f"Request {completion_id} - Streaming error: {str(e)}")
                        logger.error(f"Request {completion_id} - Traceback: {traceback.format_exc()}")
                        # Send error chunk
                        error_chunk = {
                            'id': completion_id,
                            'object': 'error',
                            'error': {'message': str(e), 'type': type(e).__name__}
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        raise
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                # Non-streaming response
                logger.info(f"Request {completion_id} - Starting non-streaming response")
                model_call_start = time.time()
                
                result = model.chat.remote(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                model_call_time = time.time() - model_call_start
                logger.info(f"Request {completion_id} - Model call completed in {model_call_time:.3f}s")
                
                # Log response details
                response_content = result["choices"][0]["message"]["content"]
                content_length = len(response_content)
                content_preview = response_content[:100] + "..." if content_length > 100 else response_content
                logger.debug(
                    f"Request {completion_id} - Response: length={content_length}, "
                    f"preview='{content_preview}'"
                )
                
                response_data = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": request.model,
                    "choices": result["choices"],
                    "usage": {
                        "prompt_tokens": -1,  # vLLM can provide this
                        "completion_tokens": -1,
                        "total_tokens": -1,
                    }
                }
                
                total_time = time.time() - request_start
                logger.info(
                    f"Request {completion_id} - Completed successfully in {total_time:.3f}s "
                    f"(model: {model_call_time:.3f}s), response length: {content_length} chars"
                )
                
                return response_data
        except HTTPException:
            # Re-raise HTTP exceptions (like auth errors) without modification
            raise
        except Exception as e:
            total_time = time.time() - request_start
            logger.error(f"Request {completion_id} - Error after {total_time:.3f}s: {str(e)}")
            logger.error(f"Request {completion_id} - Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating completion: {str(e)}"
            )
    
    @web_app.post("/v1/completions")
    async def completions(request: CompletionRequest, authorization: str = Header(None)):
        """Text completions endpoint (OpenAI compatible)."""
        request_start = time.time()
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        # Log request details
        prompt_preview = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
        logger.info(
            f"POST /v1/completions - Request ID: {completion_id} - "
            f"Model: {request.model}, Prompt length: {len(request.prompt)}, "
            f"Max tokens: {request.max_tokens}, Temperature: {request.temperature}"
        )
        logger.debug(f"Request {completion_id} - Prompt preview: '{prompt_preview}'")
        
        # Auth check
        if API_KEY and authorization != f"Bearer {API_KEY}":
            logger.warning(f"Request {completion_id} - Authentication failed")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        try:
            model = Model()
            model_call_start = time.time()
            logger.info(f"Request {completion_id} - Calling model.generate.remote()")
            
            text = model.generate.remote(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            
            model_call_time = time.time() - model_call_start
            text_length = len(text)
            text_preview = text[:100] + "..." if text_length > 100 else text
            logger.info(
                f"Request {completion_id} - Model call completed in {model_call_time:.3f}s, "
                f"response length: {text_length} chars"
            )
            logger.debug(f"Request {completion_id} - Response preview: '{text_preview}'")
            
            response_data = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "text": text,
                    "index": 0,
                    "finish_reason": "stop",
                }]
            }
            
            total_time = time.time() - request_start
            logger.info(
                f"Request {completion_id} - Completed successfully in {total_time:.3f}s "
                f"(model: {model_call_time:.3f}s)"
            )
            
            return response_data
        except Exception as e:
            total_time = time.time() - request_start
            logger.error(f"Request {completion_id} - Error after {total_time:.3f}s: {str(e)}")
            logger.error(f"Request {completion_id} - Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating completion: {str(e)}"
            )
    
    @web_app.get("/health")
    async def health():
        """Health check endpoint."""
        logger.debug("GET /health - Health check requested")
        return {"status": "ok"}
    
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
