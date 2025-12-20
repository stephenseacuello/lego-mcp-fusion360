# LEGO MCP Dashboard - Docker Image
# Flask web dashboard for LEGO brick management
#
# Build from project root with:
#   docker build -f dashboard/Dockerfile -t lego-mcp-dashboard .
#
# Or use docker-compose from project root:
#   docker-compose -f dashboard/docker-compose.yml up

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    FLASK_APP=app.py \
    PYTHONPATH=/app:/app/shared:/app/tools

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (minimal set for dashboard)
RUN pip install --no-cache-dir \
    Flask>=2.3.0 \
    Flask-SocketIO>=5.3.0 \
    python-socketio>=5.8.0 \
    python-engineio>=4.5.0 \
    requests>=2.28.0 \
    gunicorn>=21.0.0 \
    eventlet>=0.33.0 \
    python-dotenv>=1.0.0

# Copy dashboard application
COPY dashboard/ /app/

# Copy shared modules and tools (from project root context)
COPY shared/ /app/shared/
COPY mcp-server/src/tools/ /app/tools/

# Create output directory
RUN mkdir -p /output

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "app:create_app()"]
