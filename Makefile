.PHONY: help install install-dev install-vision test lint format run-dashboard run-slicer docker-up docker-down clean

help:
	@echo "LEGO MCP Fusion360 - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install core dependencies"
	@echo "  make install-dev    Install dev dependencies (linting, testing)"
	@echo "  make install-vision Install vision/ML dependencies"
	@echo "  make install-all    Install everything"
	@echo ""
	@echo "Development:"
	@echo "  make run-dashboard  Start Flask dashboard on :5000"
	@echo "  make run-slicer     Start slicer service on :8081"
	@echo "  make run-mcp        Start MCP server"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-e2e       Run end-to-end tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Check code with ruff"
	@echo "  make format         Format code with black"
	@echo "  make check          Run lint + format check"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      Start all services in Docker"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make docker-build   Build Docker images"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove cache and build artifacts"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio black ruff pre-commit

install-vision:
	pip install -r requirements-vision.txt

install-all: install install-dev install-vision

# Run services
run-dashboard:
	cd dashboard && python app.py

run-slicer:
	cd slicer-service && python -m uvicorn src.slicer_api:app --host 0.0.0.0 --port 8081 --reload

run-mcp:
	cd mcp-server && python -m src.server

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/test_lego_specs.py -v

test-e2e:
	pytest tests/test_e2e.py -v --tb=short

test-vision:
	pytest tests/test_phase2_digital_twin.py -v

# Code quality
lint:
	ruff check shared/ dashboard/ tests/

format:
	black shared/ dashboard/ tests/ mcp-server/src/ slicer-service/src/

format-check:
	black --check shared/ dashboard/ tests/

check: lint format-check

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
