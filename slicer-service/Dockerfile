# CuraEngine Slicer Service for LEGO MCP
# Provides HTTP API for slicing STL files to G-code
# Uses CuraEngine which has ARM64 support (works on Apple Silicon)

FROM python:3.11-slim

LABEL maintainer="LegoMCP Project"
LABEL description="CuraEngine CLI service for 3D print G-code generation"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and CuraEngine
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    cura-engine \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart

# Create directories
RUN mkdir -p /app/profiles /output/gcode/3dprint

# Copy application files
COPY src/ /app/
COPY profiles/ /app/profiles/

WORKDIR /app

# Expose API port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8766/health || exit 1

# Run the API server
CMD ["uvicorn", "slicer_api:app", "--host", "0.0.0.0", "--port", "8766"]
