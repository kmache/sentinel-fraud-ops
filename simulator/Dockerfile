# Use python 3.10 slim
FROM python:3.10-slim

# CRITICAL: Prevents Python from buffering stdout/stderr
# Without this, you won't see "Sent MOCK..." logs in real-time
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy the producer script
COPY . .

# Data is mounted via volume in docker-compose, so we don't copy it here.

CMD ["python", "producer.py"]