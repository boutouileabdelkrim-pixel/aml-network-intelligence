# =========================================================
# AML Network Intelligence — Makefile
# =========================================================

PYTHON := python3.11
UV     := uv

.DEFAULT_GOAL := help

# ---------------------------------------------------------
# Help
# ---------------------------------------------------------
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make setup        Install dependencies and pre-commit hooks"
	@echo "  make data         Download raw dataset from Kaggle"
	@echo "  make clean-data   Run data cleaning pipeline"
	@echo "  make features     Build tabular + graph features"
	@echo "  make train        Train all models with MLflow tracking"
	@echo "  make api          Run FastAPI locally"
	@echo "  make mlflow       Run MLflow UI locally"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run ruff + mypy"
	@echo "  make format       Format code with ruff"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start docker-compose stack"
	@echo "  make docker-down  Stop docker-compose stack"
	@echo "  make clean        Remove caches and build artifacts"

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
.PHONY: setup
setup:
	$(UV) sync --extra dev
	$(UV) run pre-commit install || true
	cp -n .env.example .env || true
	@echo "✅ Setup done. Edit .env and run 'make data'."

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
.PHONY: data
data:
	$(UV) run python -m src.data.download

.PHONY: clean-data
clean-data:
	$(UV) run python -m src.data.clean

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------
.PHONY: features
features:
	$(UV) run python -m src.features.build

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
.PHONY: train
train:
	$(UV) run python -m src.models.train

# ---------------------------------------------------------
# API & MLflow
# ---------------------------------------------------------
.PHONY: api
api:
	$(UV) run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: mlflow
mlflow:
	$(UV) run mlflow ui --backend-store-uri ./mlruns --port 5000

# ---------------------------------------------------------
# Tests & Quality
# ---------------------------------------------------------
.PHONY: test
test:
	$(UV) run pytest

.PHONY: lint
lint:
	$(UV) run ruff check src tests
	$(UV) run mypy src

.PHONY: format
format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

# ---------------------------------------------------------
# Docker
# ---------------------------------------------------------
.PHONY: docker-build
docker-build:
	docker build -t aml-network-intelligence:latest -f docker/Dockerfile .

.PHONY: docker-up
docker-up:
	docker compose -f docker/docker-compose.yml up --build

.PHONY: docker-down
docker-down:
	docker compose -f docker/docker-compose.yml down

# ---------------------------------------------------------
# Clean
# ---------------------------------------------------------
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info
