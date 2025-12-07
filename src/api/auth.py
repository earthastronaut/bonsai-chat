"""
Authentication utilities.
"""

import os
import logging
from fastapi import HTTPException, Header

logger = logging.getLogger("llm-api")


def get_api_key() -> str | None:
    """Get API key from environment."""
    return os.environ.get("API_KEY")


def verify_auth(authorization: str | None, api_key: str | None, request_id: str) -> None:
    """
    Verify authentication token.
    
    Args:
        authorization: Authorization header value
        api_key: API key from environment
        request_id: Request ID for logging
        
    Raises:
        HTTPException: If authentication fails
    """
    auth_required = api_key is not None
    auth_provided = authorization is not None
    
    if auth_required:
        if not auth_provided:
            logger.warning(f"Request {request_id} - Authentication failed: No authorization header provided")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        auth_valid = authorization == f"Bearer {api_key}"
        if not auth_valid:
            logger.warning(
                f"Request {request_id} - Authentication failed: Invalid API key "
                f"(provided: {authorization[:20] if authorization else None}...)"
            )
            raise HTTPException(status_code=401, detail="Invalid API key")
        logger.info(f"Request {request_id} - Authentication successful")
    else:
        logger.debug(f"Request {request_id} - No authentication required (API_KEY not set)")

