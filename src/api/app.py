"""
FastAPI application setup.
"""

import logging
from fastapi import FastAPI

from .middleware import setup_cors, setup_request_logging
from .routes import setup_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_app(Model):
    """
    Create and configure FastAPI application.
    
    Args:
        Model: The Modal Model class to use for inference
        
    Returns:
        FastAPI application instance
    """
    web_app = FastAPI(title="LLM API")
    
    # Setup middleware
    setup_cors(web_app)
    setup_request_logging(web_app)
    
    # Setup routes
    setup_routes(web_app, Model)
    
    return web_app

