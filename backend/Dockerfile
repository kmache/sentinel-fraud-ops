FROM python:3.12-slim

WORKDIR /app

# 1. Install System Deps
RUN apt update && apt install -y gcc curl libgomp1 && rm -rf /var/lib/apt/lists/*

# 2. Install The Sentinel Library (src)
# We copy setup.py and src first to cache this layer
COPY setup.py .
COPY src/ ./src/
RUN pip install --no-cache-dir .

# 3. Install Backend Dependencies
COPY backend/requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

# 4. Copy Backend Code
COPY backend/ .

# 5. Copy Production Models (Baked into image)
COPY models/prod_v1 /app/models/prod_v1

# 6. Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


