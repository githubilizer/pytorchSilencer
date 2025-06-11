#!/usr/bin/env python3
"""
PyTorch Silencer - LSTM Demo
A simple command-line application to demonstrate the LSTM-based silence detection
"""

import os
import sys
import argparse
import numpy as np
import torch
from data_processor import TranscriptProcessor
from silence_model import SilencePredictor

def train_model(data_dir, model_path, sequence_length=10, epochs=50, batch_size=8):
    """Train an LSTM model on transcript data"""
    print(f"Training model with sequence length {sequence_length}, {epochs} epochs")
    
    # Create LSTM predictor
    predictor = SilencePredictor(sequence_length=sequence_length)
    
    # Load data as sequences
    print(f"Loading training data from {data_dir}")
    features, labels = TranscriptProcessor.load_training_sequences(
        data_dir, sequence_length=sequence_length, stride=1
    )
    
    if len(features) == 0:
        print("Error: No valid training data found.")
        return False
    
    print(f"Loaded {len(features)} training sequences")
    
    # Train model
    print("Training model...")
    losses = predictor.train(features, labels, epochs=epochs, batch_size=batch_size)
    
    # Plot losses if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(model_path + ".loss.png")
        print(f"Loss curve saved to {model_path}.loss.png")
    except ImportError:
        pass
    
    # Save model
    print(f"Saving model to {model_path}")
    predictor.save_model(model_path)
    
    print("Training complete!")
    return True

def process_transcript(model_path, input_file, output_file, threshold=0.5):
    """Process a transcript file using the trained LSTM model"""
    print(f"Loading model from {model_path}")
    predictor = SilencePredictor(model_path=model_path)
    
    print(f"Reading transcript from {input_file}")
    transcript = TranscriptProcessor.parse_transcript(input_file)
    
    print(f"Extracting features...")
    features = transcript.to_features()
    
    if len(features) == 0:
        print("Error: No valid features found in transcript.")
        return False
    
    print(f"Making predictions with threshold {threshold}...")
    keep_predictions = predictor.predict(features, threshold)
    cut_predictions = [not p for p in keep_predictions]  # Invert to get cut markers
    
    print(f"Saving processed transcript to {output_file}")
    TranscriptProcessor.save_processed_transcript(transcript, output_file, cut_predictions)
    
    # Print summary
    total_silences = sum(1 for i, entry in enumerate(transcript.entries[:-1]) if 
                         transcript.entries[i+1].start_time - entry.end_time > 0.01)
    cut_silences = sum(cut_predictions[:total_silences])
    
    print(f"Total silences: {total_silences}")
    print(f"Silences marked for cutting: {cut_silences} ({cut_silences/max(1, total_silences)*100:.1f}%)")
    
    print("Done!")
    return True

def main():
    parser = argparse.ArgumentParser(description="LSTM-based silence detection for transcripts")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data", "-d", required=True, help="Directory with training data")
    train_parser.add_argument("--model", "-m", required=True, help="Path to save model")
    train_parser.add_argument("--seq-length", "-s", type=int, default=10, help="Sequence length")
    train_parser.add_argument("--epochs", "-e", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    
    # Process parser
    process_parser = subparsers.add_parser("process", help="Process a transcript")
    process_parser.add_argument("--model", "-m", required=True, help="Path to trained model file")
    process_parser.add_argument("--input", "-i", required=True, help="Input transcript file")
    process_parser.add_argument("--output", "-o", help="Output transcript file (default: input_cleaned.txt)")
    process_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Prediction threshold (0-1)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.data, args.model, args.seq_length, args.epochs, args.batch_size)
    elif args.command == "process":
        # Set default output path if not provided
        if not args.output:
            base_name, ext = os.path.splitext(args.input)
            args.output = f"{base_name}_cleaned{ext}"
        
        process_transcript(args.model, args.input, args.output, args.threshold)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 