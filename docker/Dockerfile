# =========================================================
# AML Network Intelligence — API Docker image
# Multi-stage: build with uv, run minimal
# =========================================================

FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml README.md ./

# Install deps into a venv
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        fastapi uvicorn[standard] pydantic pydantic-settings \
        loguru python-dotenv pyyaml \
        numpy pandas pyarrow scikit-learn xgboost

# =========================================================
# Runtime stage
# =========================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/
COPY models/xgb_final.json ./models/xgb_final.json
COPY models/xgb_final_features.pkl ./models/xgb_final_features.pkl

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Healthcheck (works once API exists in Phase 9)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
