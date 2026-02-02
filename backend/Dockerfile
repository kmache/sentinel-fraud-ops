FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Install Backend-specific requirements (FastAPI, etc.)
COPY backend/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# 2. Install the 'sentinel' package
COPY pyproject.toml .
COPY src/ src/
RUN uv pip install --system --no-cache -e .

# 3. Copy Backend code
COPY backend/main.py backend/schemas.py ./

EXPOSE 8000

# Run from the directory where main.py exists
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]