"""
Source package for Modal LLM backend.

This package follows Modal's recommended structure:
- app.py defines the Modal App and all functions
- All modules are imported here to ensure decorators are registered
- Deploy with: modal deploy -m src.app
"""

# Import all modules to ensure @modal.method() and other decorators are registered
from . import app
from . import config
from . import images
from . import api

__all__ = ["app", "config", "images", "api"]

