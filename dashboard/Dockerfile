# Use Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout/stderr (Crucial for Docker logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential: for compiling python packages
# curl: for healthcheck
# libgomp1: required for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy application code (including app.py)
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

USER streamlit

# Expose Streamlit port
EXPOSE 8501

# Health check (Streamlit specific endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit pointing to app.py
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
