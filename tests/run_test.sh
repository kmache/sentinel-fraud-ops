#!/bin/bash
# Sentinel Fraud Detection System - Test Runner

# Ensure the script stops on the first error
set -e

# Navigate to project root relative to this script
cd "$(dirname "$0")/.."

# Export PYTHONPATH so imports work from root
export PYTHONPATH=$PYTHONPATH:.

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed. Please run 'pip install pytest httpx pytest-asyncio'"
    exit 1
fi

echo "--- Sentinel Fraud Detection Test Suite ---"

case "$1" in
  api)
    echo "Running API (FastAPI) tests..."
    python3 -m pytest tests/test_api.py -v
    ;;
  preprocessing)
    echo "Running Data Pipeline & ML Evaluation tests..."
    python3 -m pytest tests/test_preprocessing.py -v
    ;;
  coverage)
    echo "Generating code coverage report..."
    python3 -m pytest tests/ --cov=src --cov=backend --cov-report=html
    echo "Done! View report at: htmlcov/index.html"
    ;;
  *)
    echo "Running all tests..."
    python3 -m pytest tests/ -v
    ;;
esac

echo "--- Tests Complete ---"