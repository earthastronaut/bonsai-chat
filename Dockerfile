# Open WebUI - LM Studio-like chat interface
# Connects to Modal GPU backend for inference

FROM ghcr.io/open-webui/open-webui:main

# Railway sets PORT env var, Open WebUI expects 8080
ENV PORT=8080

# These will be overridden by Railway environment variables
ENV WEBUI_AUTH=true
ENV WEBUI_NAME="LLM Studio"
ENV ENABLE_SIGNUP=true

# Data persistence (attach Railway volume to /app/backend/data)
# VOLUME ["/app/backend/data"]

EXPOSE 8080

