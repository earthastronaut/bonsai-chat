"""
FastAPI middleware setup.
"""

import time
import uuid
import json
import logging
import traceback
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("llm-api")


def setup_cors(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_request_logging(app):
    """Setup request logging middleware."""
    @app.middleware("http")
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

