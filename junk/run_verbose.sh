#!/bin/bash

# Verbose launcher for PyTorch Silencer
# This script enables maximum logging and verbosity for debugging

# Set environment variables for verbosity
export PYTHONUNBUFFERED=1
export TORCH_CPP_LOG_LEVEL=INFO

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if ! python -c "import torch, PyQt5" &>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application with debugging info
echo "Starting PyTorch Silencer with verbose output..."
python app.py 