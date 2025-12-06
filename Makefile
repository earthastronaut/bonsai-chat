.PHONY: help install venv check-modal check-auth setup-info deploy local-dev clean

# Default target
help:
	@echo "🚀 Cloud LLM Studio - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Set up virtual environment and install dependencies"
	@echo "  make venv         - Create virtual environment only"
	@echo "  make check-modal  - Check if Modal CLI is installed"
	@echo "  make check-auth   - Check Modal authentication"
	@echo "  make setup-info   - Show setup instructions"
	@echo "  make deploy       - Deploy the GPU backend to Modal"
	@echo "  make local-dev    - Run local development with docker compose"
	@echo "  make clean        - Remove virtual environment"
	@echo ""

# Virtual environment path
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Create virtual environment
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "📦 Creating virtual environment..."; \
		python3 -m venv $(VENV); \
		echo "✅ Virtual environment created"; \
	else \
		echo "✅ Virtual environment already exists"; \
	fi

# Install dependencies
install: venv
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"
	@echo ""
	@echo "💡 Activate the virtual environment with:"
	@echo "   source $(VENV)/bin/activate"

# Check for Modal CLI
check-modal: venv
	@echo "🔍 Checking for Modal CLI..."
	@if $(PYTHON) -m modal --version &> /dev/null 2>&1 || command -v modal &> /dev/null; then \
		echo "✅ Modal CLI is installed"; \
	else \
		echo "❌ Modal CLI not found. Installing..."; \
		$(PIP) install modal; \
		echo "✅ Modal CLI installed"; \
	fi

# Check Modal authentication
check-auth: check-modal
	@echo "🔐 Checking Modal authentication..."
	@if modal token list &> /dev/null 2>&1; then \
		echo "✅ Modal authentication configured"; \
	else \
		echo "⚠️  Modal authentication not configured"; \
		echo "   Run: modal setup"; \
	fi

# Show setup instructions
setup-info:
	@echo "📋 Setup Steps:"
	@echo ""
	@echo "1. Create API key secret (for auth between frontend & backend):"
	@echo "   modal secret create api-auth API_KEY=your-secret-key-here"
	@echo ""
	@echo "2. (Optional) Add HuggingFace token for gated models:"
	@echo "   modal secret create huggingface HF_TOKEN=hf_xxxxx"
	@echo ""
	@echo "3. Deploy the GPU backend:"
	@echo "   make deploy"
	@echo "   or: modal deploy modal_backend.py"
	@echo ""
	@echo "4. After deploy, you'll get a URL like:"
	@echo "   https://YOUR_USERNAME--llm-backend-api.modal.run"
	@echo ""
	@echo "5. Deploy Open WebUI to Railway:"
	@echo ""
	@echo "   a) Go to railway.app → New Project → Deploy from GitHub"
	@echo "   b) Select this repository"
	@echo "   c) Add a Volume mounted to /app/backend/data"
	@echo "   d) Set environment variables:"
	@echo "      OPENAI_API_BASE_URL = https://YOUR_USERNAME--llm-backend-api.modal.run/v1"
	@echo "      OPENAI_API_KEY = your-secret-key-here"
	@echo "      WEBUI_AUTH = true"
	@echo "      WEBUI_NAME = LLM Studio"
	@echo "      ENABLE_SIGNUP = true"
	@echo "      PORT = 8080"
	@echo ""
	@echo "   Or use the one-click template:"
	@echo "   https://railway.app/template/open-webui"
	@echo ""
	@echo "   For local development:"
	@echo "   make local-dev"
	@echo ""
	@echo "6. Open your Railway URL, create an account, and start chatting!"
	@echo ""
	@echo "💡 Tips:"
	@echo "   • The GPU scales to zero after 2 min idle (saves $$$)"
	@echo "   • First request after idle takes ~30-60 sec (cold start)"
	@echo "   • Change MODEL_ID in modal_backend.py for different models"
	@echo ""

# Deploy to Modal
deploy: check-auth
	@echo "🚀 Deploying to Modal..."
	@if command -v modal &> /dev/null; then \
		modal deploy modal_backend.py; \
	else \
		$(PYTHON) -m modal deploy modal_backend.py; \
	fi

# Run local development
local-dev:
	@echo "🐳 Starting local development with docker compose..."
	docker compose up

# Clean virtual environment
clean:
	@echo "🧹 Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "✅ Virtual environment removed"

