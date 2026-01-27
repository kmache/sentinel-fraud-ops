# Fraud Detection Live Demo

A real-time fraud detection system with machine learning models, Kafka streaming, Redis storage, and Streamlit dashboard.

## 🚀 Quick Start

### 1. Clone/Copy the Project
```bash
# Copy all the created files to the fraud-detection-demo directory
# Your directory should look like:
fraud-detection-demo/
├── docker-compose.yml
├── data/
│   └── test_raw.csv
├── models/
│   ├── cb_model.pkl
│   └── lgb_model.pkl
├── artifacts/
│   ├── sentinel_preprocessor.pkl
│   ├── sentinel_engineer.pkl
│   ├── cb_features.pkl
│   └── lgb_features.pkl
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── processor_worker.py
│   └── utils/model_pipeline.py
├── simulator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── producer.py
└── dashboard/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py

This is a comprehensive technical review of the Sentinel Fraud Detection System. This document summarizes the architecture, machine learning strategy, and engineering decisions that make this a production-grade MLOps project.

Project Review: Sentinel Fraud Detection System

"Real-Time Payment Monitoring & High-Stakes ML Inference"

1. Executive Summary

Sentinel is an end-to-end, real-time fraud detection platform designed to identify suspicious financial transactions as they occur. By combining Ensemble Learning with a high-throughput Stream Processing architecture, Sentinel bridges the gap between static data science experiments and live production systems.

2. The Architectural Stack (The "How")

The system is built as a microservices-based distributed system, containerized with Docker:

Ingestion (Kafka): Acts as the high-speed nervous system. A Python-based simulator streams the IEEE-CIS Fraud dataset into a Kafka topic, mimicking a real-world payment gateway.

Processing Brain (Worker): A heavy-lift Python service that performs on-the-fly feature engineering. It unpickles custom Sentinel classes to transform raw data into 150+ features.

State Management (Redis): Serves as an ultra-fast, in-memory store for real-time counters (Fraud counts, volume) and a "sliding window" of the most recent transactions.

Backend Interface (FastAPI): A high-performance API providing structured access to metrics and health checks.

Analytics Dashboard (Streamlit): A business-facing UI that visualizes throughput, risk distributions, and live fraud alerts for human analysts.

3. Machine Learning Strategy

Sentinel uses a Champion-Challenger Ensemble approach:

Champion: CatBoost – Highly robust with categorical data, providing the primary fraud score.

Challenger: LightGBM – Optimized for speed and gradient boosting efficiency.

Ensemble Logic: The system computes a weighted average of both models. This increases the "Area Under the Precision-Recall Curve" and reduces False Positives, which is critical in finance to avoid blocking legitimate customers.

4. Engineering "Production-Grade" Features

What separates this project from a standard Kaggle notebook are the Production Safeguards implemented:

Feature Alignment (Schema Enforcement): During inference, the worker uses .reindex() to strictly enforce the feature list and column order. This ensures the model never receives unexpected columns, a common cause of production crashes.

Graceful Degradation (Fallback Inference): If the Challenger model (LightGBM) fails due to data-type mismatches or errors, the system automatically catches the exception and falls back to the Champion (CatBoost) score, ensuring the payment stream never stops.

Type-Safe Engineering: Custom handling for NaN values and strict casting to int32 for mathematical operations prevents overflow errors and ensures consistency between Training and Inference.

Asynchronous Scalability: By using Kafka, the ingestion is decoupled from the processing. If transaction volume spikes, we can simply spin up more "Worker" containers to handle the load without changing the API code.

5. Key Metrics Managed

The system tracks business-critical KPIs in real-time:

Money Saved: Estimated loss prevention based on flagged fraud.

Fraud Rate: Percentage of total traffic identified as malicious.

Throughput: System performance measured in Transactions per Second (TPS).

Model Separation: Visual histogram showing how clearly the model distinguishes between "Legit" (scores near 0) and "Fraud" (scores near 1).

6. Project Impact (Business Value)

Latency: The pipeline processes a transaction from "Swipe" to "Score" in milliseconds.

Observability: Analysts can monitor system health via a centralized dashboard.

Stability: Dockerization ensures the "it works on my machine" problem is eliminated, allowing for seamless deployment to AWS, GCP, or Azure.

Final Assessment

The Sentinel Fraud Detection System is a sophisticated implementation of MLOps. It demonstrates a mastery of data engineering (Kafka/Redis), backend development (FastAPI), and advanced Machine Learning (Ensemble methods), making it a high-value asset for any organization requiring real-time predictive analytics.


Here is the step-by-step guide to running your Sentinel Fraud Detection System. Since you have organized everything into a Docker-friendly structure, we will use Docker Compose to orchestrate the entire stack.

Prerequisites

Docker Desktop installed and running.

Git Bash (Windows) or Terminal (Mac/Linux).

RAM: Ensure Docker has access to at least 4GB - 6GB of RAM (ML models + Kafka + Java require memory).

Step 1: Verify utils.py Consistency (Critical)

The most common error in this setup is a ModuleNotFoundError when loading the .pkl files.

Open backend/utils.py.

Ensure it contains the exact definition of the SentinelFeatureEngineering and SentinelPreprocessing classes used when you trained the models.

If you trained the models in a notebook, copy the class code from the notebook into backend/utils.py.

Step 2: Build and Start the Stack

Open your terminal inside the project root folder (where docker-compose.yml is located).

Build the Docker images:

code
Bash
download
content_copy
expand_less
docker-compose build

This will create images for api, stream-processor, producer, and dashboard.

Start the services:

code
Bash
download
content_copy
expand_less
docker-compose up -d

The -d flag runs it in "detached" mode (background).

Step 3: Monitor the Startup

Since Kafka and Zookeeper take time to initialize, the other services might restart a few times before connecting. Watch the logs to confirm stability.

Follow the Stream Processor logs (The Brain):

code
Bash
download
content_copy
expand_less
docker-compose logs -f stream-processor

Wait until you see: ✅ Kafka consumer connected and ✅ Redis connected.

Follow the Producer logs (The Simulator):
Open a new terminal tab:

code
Bash
download
content_copy
expand_less
docker-compose logs -f producer

You should see: 📤 Sent CSV: tx_12345... or 📤 Sent MOCK....

Step 4: Access the Interfaces

Once everything is running, open your web browser to access the different components.

1. 🛡️ The Dashboard (Streamlit)

URL: http://localhost:8501

What to do: Navigate through the 4 pages. You should see charts updating in real-time on "Page 2: Real-Time Operations".

2. 📡 API Documentation (FastAPI)

URL: http://localhost:8000/docs

What to do: Expand /stats/detailed and click "Try it out" -> "Execute" to see raw JSON metrics.

3. 🔎 Kafka UI (Monitoring)

URL: http://localhost:8080

What to do: Click on Topics. You should see transactions (input) and predictions (output) with message counts increasing.

4. 🧠 Redis Insight (Database View)

URL: http://localhost:8001

What to do:

Click "Add Redis Database".

Host: redis

Port: 6379

Password: sentinel_pass_2024

Explore keys like sentinel_stream (List) and stats:fraud_count (String).

Step 5: Validating the Flow

If the system is working correctly, this is the data lifecycle you are witnessing:

Producer reads data/test_raw.csv 
→
→
 Sends JSON to Kafka (transactions topic).

Stream Processor reads Kafka 
→
→
 Loads .pkl from /artifacts 
→
→
 Predicts Fraud 
→
→
 Writes to Redis & Kafka (predictions).

Dashboard polls Redis every few seconds 
→
→
 Updates Charts.

Backend API queries Redis 
→
→
 Serves health checks.

Step 6: stopping the System

To stop all containers and remove the networks (data in Redis volumes will persist):

code
Bash
download
content_copy
expand_less
docker-compose down

To stop and delete volume data (reset Redis counters):

code
Bash
download
content_copy
expand_less
docker-compose down -v
🚨 Troubleshooting Common Errors

1. ModuleNotFoundError: No module named 'utils' in Processor logs

Cause: The pickle file saved the class path as utils, but the script can't find it.

Fix: Ensure the sys.modules hack is present in backend/processor_worker.py:

code
Python
download
content_copy
expand_less
import utils
sys.modules['sentinel_fraud_detection'] = utils # Or whatever name the notebook had

2. NoBrokersAvailable

Cause: Kafka isn't ready yet.

Fix: Wait 30 seconds. The Python scripts have retry logic (retries=30) and will connect eventually.

3. Dashboard is Empty

Cause: Redis keys don't match.

Fix: Ensure processor_worker.py writes to sentinel_stream and app.py reads from sentinel_stream. (The code provided in previous steps handles this).