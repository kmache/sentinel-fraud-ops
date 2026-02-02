FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for ML (LightGBM/XGBoost need libgomp1)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Install Worker-specific requirements (Kafka, Redis, etc.)
COPY worker/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# 2. Install the 'sentinel' package from the root pyproject.toml
# This makes 'import sentinel' work everywhere
COPY pyproject.toml .
COPY src/ src/
RUN uv pip install --system --no-cache -e .

# 3. Copy Worker script
COPY worker/processor.py .

# Models are mounted via volume in docker-compose
CMD ["python", "processor.py"]