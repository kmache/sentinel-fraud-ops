FROM python:3.12-slim

WORKDIR /app

# 1. System Deps
RUN apt-get update && apt-get install -y gcc libgomp1 && rm -rf /var/lib/apt/lists/*

# 2. Install Sentinel Library
COPY setup.py .
COPY src/ ./src/
RUN pip install --no-cache-dir .

# 3. Install Worker Deps
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Worker Code
COPY worker/ .

# 5. Copy Model Bundle
COPY models/prod_v1 /app/models/prod_v1

CMD ["python", "processor_worker.py"]