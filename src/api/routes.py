"""
API route handlers.
"""

import time
import uuid
import json
import logging
import traceback
from fastapi import HTTPException, Header, Request
from fastapi.responses import StreamingResponse

from .models import ChatRequest, CompletionRequest
from .auth import get_api_key, verify_auth
from ..config import MODEL_ID

logger = logging.getLogger("llm-api")


def setup_routes(app, Model):
    """Setup API routes."""
    
    @app.get("/v1/models")
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
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest, authorization: str = Header(None)):
        """Chat completions endpoint (OpenAI compatible)."""
        request_start = time.time()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        api_key = get_api_key()
        
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
        verify_auth(authorization, api_key, completion_id)
        
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
                        try:
                            chunks = model.chat_stream.remote(
                                messages=messages,
                                max_tokens=request.max_tokens,
                                temperature=request.temperature,
                            )
                            # Ensure chunks is iterable
                            if chunks is None:
                                raise ValueError("Model returned None chunks")
                            model_call_time = time.time() - model_call_start
                            logger.info(
                                f"Request {completion_id} - Model call completed in {model_call_time:.3f}s, "
                                f"received {len(chunks) if chunks else 0} chunks"
                            )
                        except Exception as model_error:
                            model_call_time = time.time() - model_call_start
                            logger.error(
                                f"Request {completion_id} - Model call failed after {model_call_time:.3f}s: {str(model_error)}"
                            )
                            raise
                        
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
    
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, authorization: str = Header(None)):
        """Text completions endpoint (OpenAI compatible)."""
        request_start = time.time()
        completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        api_key = get_api_key()
        
        # Log request details
        prompt_preview = request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
        logger.info(
            f"POST /v1/completions - Request ID: {completion_id} - "
            f"Model: {request.model}, Prompt length: {len(request.prompt)}, "
            f"Max tokens: {request.max_tokens}, Temperature: {request.temperature}"
        )
        logger.debug(f"Request {completion_id} - Prompt preview: '{prompt_preview}'")
        
        # Auth check
        verify_auth(authorization, api_key, completion_id)
        
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
        except HTTPException:
            raise
        except Exception as e:
            total_time = time.time() - request_start
            logger.error(f"Request {completion_id} - Error after {total_time:.3f}s: {str(e)}")
            logger.error(f"Request {completion_id} - Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating completion: {str(e)}"
            )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        logger.debug("GET /health - Health check requested")
        return {"status": "ok"}

