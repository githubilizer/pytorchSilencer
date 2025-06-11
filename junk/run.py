#!/usr/bin/env python3
"""
PyTorch Silencer - All-in-one launcher
This script handles all setup and launches the appropriate version of the application.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Application modes
GUI_MODE = "gui"
CLI_MODE = "cli"
SIMPLE_MODE = "simple"

def setup_environment():
    """Set up virtual environment and install dependencies if needed"""
    venv_dir = Path("venv")
    
    # Check if virtual environment exists
    if not venv_dir.exists():
        print("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError:
            print("Failed to create virtual environment. Make sure Python 3 is installed with venv support.")
            return False
    
    # Get paths to Python and pip in virtual environment
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"
    
    # Check if dependencies are installed
    try:
        result = subprocess.run(
            [str(venv_python), "-c", "import torch, PyQt5, matplotlib, numpy, sklearn"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print("Installing required packages...")
            subprocess.run([str(venv_pip), "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")
        return False
    
    return str(venv_python)

def run_gui_app(python_path):
    """Run the PyQt GUI application"""
    print("Starting PyTorch Silencer GUI application...")
    try:
        subprocess.run([python_path, "app.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Error running GUI application.")
        return False

def run_cli_app(python_path, args):
    """Run the command-line LSTM demo application"""
    print("Starting PyTorch Silencer CLI application...")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Check if we need to train a model first
    model_path = args.model if args.model else "models/lstm_model.pt"
    
    if args.train or not os.path.exists(model_path):
        if not args.data:
            print("No training data specified. Using sample data...")
            args.data = "sample_data"
        
        # Train the model
        print(f"Training model with sequence length {args.seq_length}, {args.epochs} epochs...")
        train_cmd = [
            python_path, "lstm_demo.py", "train",
            "--data", args.data,
            "--model", model_path,
            "--seq-length", str(args.seq_length),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size)
        ]
        
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Error training model.")
            return False
    
    # Process transcript if provided
    if args.input:
        output_path = args.output if args.output else None
        
        process_cmd = [
            python_path, "lstm_demo.py", "process",
            "--model", model_path,
            "--input", args.input
        ]
        
        if output_path:
            process_cmd.extend(["--output", output_path])
        
        if args.threshold:
            process_cmd.extend(["--threshold", str(args.threshold)])
        
        try:
            subprocess.run(process_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Error processing transcript.")
            return False
    
    return True

def run_simple_app(python_path, args):
    """Run the simple duration-based silence cutter"""
    print("Starting simple silence cutter...")
    
    if not args.input:
        print("Error: No input file specified.")
        return False
    
    output_path = args.output if args.output else None
    threshold = args.threshold if args.threshold else 1.5
    
    cmd = [
        python_path, "simple_silence_cutter.py",
        "--input", args.input,
        "--threshold", str(threshold)
    ]
    
    if output_path:
        cmd.extend(["--output", output_path])
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Error running simple silence cutter.")
        return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch Silencer - Transcript Silence Processing")
    
    # Mode selection
    parser.add_argument("--mode", choices=[GUI_MODE, CLI_MODE, SIMPLE_MODE], default=GUI_MODE,
                       help="Application mode: gui, cli, or simple (default: gui)")
    
    # Common parameters
    parser.add_argument("--input", "-i", help="Input transcript file path")
    parser.add_argument("--output", "-o", help="Output transcript file path")
    parser.add_argument("--threshold", "-t", type=float, help="Prediction threshold (0-1) or silence duration threshold (seconds)")
    
    # LSTM parameters
    parser.add_argument("--model", "-m", help="Path to LSTM model file")
    parser.add_argument("--data", "-d", help="Path to training data directory")
    parser.add_argument("--train", action="store_true", help="Force model training even if model exists")
    parser.add_argument("--seq-length", type=int, default=10, help="Sequence length for LSTM (default: 10)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8)")
    
    args = parser.parse_args()
    
    # Create application directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("sample_data", exist_ok=True)
    
    # Set up environment
    python_path = setup_environment()
    if not python_path:
        sys.exit(1)
    
    # Run application in appropriate mode
    if args.mode == GUI_MODE:
        success = run_gui_app(python_path)
        if not success and not args.input:
            print("Falling back to CLI mode...")
            args.mode = CLI_MODE
    
    if args.mode == CLI_MODE:
        success = run_cli_app(python_path, args)
    elif args.mode == SIMPLE_MODE:
        success = run_simple_app(python_path, args)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 