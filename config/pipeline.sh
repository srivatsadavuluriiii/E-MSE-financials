#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running from: $PROJECT_ROOT"

if [ ! -f "datasets/train.csv" ]; then
    echo " Error: datasets/train.csv not found"
    exit 1
fi

echo ""
python3 scripts/data.py

echo ""
python3 scripts/train.py

echo ""
python3 scripts/predict.py

echo ""
python3 scripts/visuals.py

echo ""
python3 scripts/storePrediction.py

echo ""
python3 scripts/visuals3D.py

