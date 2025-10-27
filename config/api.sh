#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ ! -f "models/cashflow_model.pkl" ]; then
    echo " Error: Model not found"
    exit 1
fi

echo ""
echo "http://localhost:8000"
echo "http://localhost:8000/docs"

cd api && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

