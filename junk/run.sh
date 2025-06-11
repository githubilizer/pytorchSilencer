#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 to run this application."
    exit 1
fi

# Check if required packages are installed
if ! python3 -c "import torch, PyQt5" &> /dev/null; then
    echo "Installing required packages..."
    python3 -m pip install -r requirements.txt
fi

# Run the application
python3 gui_wrapper.py 