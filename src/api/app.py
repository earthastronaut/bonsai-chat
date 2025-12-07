"""
FastAPI application setup.
"""
from fastapi import FastAPI

from .middleware import setup_cors, setup_request_logging
from .routes import setup_routes



def create_app(model_registry):
    """
    Create and configure FastAPI application.
    
    Args:
        model_registry: Dictionary mapping model_id to Modal Model class
        
    Returns:
        FastAPI application instance
    """
    web_app = FastAPI(title="LLM API")
    
    # Setup middleware
    setup_cors(web_app)
    setup_request_logging(web_app)
    
    # Setup routes
    setup_routes(web_app, model_registry)
    
    return web_app

