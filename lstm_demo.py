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
import time

def train_model(data_dir, model_path, sequence_length=10, epochs=50, batch_size=8):
    """Train a new silence prediction model"""
    print(f"Training model with sequence length {sequence_length}, {epochs} epochs, batch size {batch_size}")
    
    try:
        # Load training data
        features, labels = TranscriptProcessor.load_training_sequences(
            data_dir, 
            sequence_length=sequence_length,
            stride=1
        )
        
        if len(features) == 0:
            print("No training data found. Please check the data directory.")
            return False
            
        print(f"Loaded {len(features)} training sequences with shape {features.shape}")
        
        # Initialize model - Use GPU with the nightly build that supports RTX 5060
        # Derive long silence threshold from training data
        silence_vals = features[:, :, 2].reshape(-1)
        silence_vals = silence_vals[silence_vals > 0]
        long_thresh = float(np.percentile(silence_vals, 95)) if len(silence_vals) > 0 else None
        if long_thresh is not None:
            print(f"Derived long silence threshold: {long_thresh:.2f}s")

        predictor = SilencePredictor(
            sequence_length=sequence_length,
            force_cpu=False,
            long_silence_threshold=long_thresh,
        )
        
        # Train model with more verbose output
        for epoch in range(1, epochs+1):
            print(f"Starting Epoch {epoch}/{epochs}...")
            epoch_start_time = time.time()
            
            # Train for one epoch
            epoch_losses = predictor.train(features, labels, epochs=1, batch_size=batch_size)
            
            # Print detailed progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s - Loss: {epoch_losses[0]:.6f}")
            
            # Print progress every 10% of epochs
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"Progress: {int(epoch/epochs*100)}% complete")
        
        # Save model
        predictor.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        return True
    except Exception as e:
        import traceback
        print(f"Training error: {str(e)}")
        print(traceback.format_exc())
        return False

def process_transcript(
    model_path,
    input_path,
    output_path=None,
    threshold=0.5,
    duration_threshold=None,
    keep_ratio=0.5,
):
    """Process a transcript using a trained model

    Parameters
    ----------
    model_path : str
        Path to the trained model.
    input_path : str
        Transcript file to process.
    output_path : str, optional
        Where to save the processed transcript.
    threshold : float, optional
        Prediction threshold for the LSTM model.
    duration_threshold : float, optional
        Override for long silence detection. If None, the value
        stored in the model will be used.
    keep_ratio : float, optional
        Fraction of each cut silence to keep in the output. The
        remaining duration will vary with the original silence
        length.
    """
    try:
        # Load the model - Use GPU with the nightly build that supports RTX 5060
        predictor = SilencePredictor(model_path=model_path, force_cpu=False)
        
        # Load transcript
        print(f"Loading transcript from {input_path}")
        transcript = TranscriptProcessor.parse_transcript(input_path)
        
        if len(transcript.entries) == 0:
            print("No entries found in transcript.")
            return False
        
        # Extract features
        features = transcript.to_features()
        
        if len(features) == 0:
            print("No features extracted from transcript.")
            return False
        
        # Predict which silences to cut using the model
        print("Predicting silences to cut...")
        keep_silence = predictor.predict(features, threshold=threshold)

        # Convert keep predictions to cut markers for saving
        cut_markers = [not keep for keep in keep_silence]

        # Fallback: enforce cutting of very long silences
        effective_threshold = duration_threshold
        if effective_threshold is None:
            effective_threshold = predictor.long_silence_threshold

        if effective_threshold is not None:
            for i, entry in enumerate(transcript.entries[:-1]):
                silence_duration = transcript.entries[i+1].start_time - entry.end_time
                if silence_duration >= effective_threshold:
                    cut_markers[i] = True
        
        # Calculate statistics using final cut markers
        total_silences = len(cut_markers)
        silences_to_cut = sum(1 for cut in cut_markers if cut)
        cut_percentage = silences_to_cut / total_silences * 100 if total_silences > 0 else 0
        
        print(f"Total silences: {total_silences}")
        print(f"Silences marked for cutting: {silences_to_cut} ({cut_percentage:.1f}%)")
        
        # Save processed transcript if output path provided
        if output_path:
            print(f"Saving processed transcript to {output_path}")
            remaining = []
            for i, entry in enumerate(transcript.entries[:-1]):
                silence_duration = transcript.entries[i + 1].start_time - entry.end_time
                if cut_markers[i]:
                    remain = max(silence_duration * keep_ratio, 0.0)
                    remaining.append(remain)
                else:
                    remaining.append(0.0)
            TranscriptProcessor.save_processed_transcript(
                transcript, output_path, cut_markers, remaining
            )
        
        return True
    except Exception as e:
        import traceback
        print(f"Processing error: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LSTM-based silence detection for transcripts")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data", required=True, help="Path to directory with training data")
    train_parser.add_argument("--model", required=True, help="Path to save model")
    train_parser.add_argument("--seq-length", type=int, default=10, help="Sequence length for LSTM")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    
    # Process subcommand
    process_parser = subparsers.add_parser("process", help="Process a transcript")
    process_parser.add_argument("--model", required=True, help="Path to trained model")
    process_parser.add_argument("--input", required=True, help="Input transcript file")
    process_parser.add_argument("--output", help="Output transcript file")
    process_parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold (0-1)")
    process_parser.add_argument(
        "--duration-threshold", type=float, default=None,
        help="Override automatic long silence threshold (seconds)")
    process_parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of each cut silence to keep (0-1)",
    )
    
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == "train":
        train_model(args.data, args.model, args.seq_length, args.epochs, args.batch_size)
    elif args.command == "process":
        process_transcript(
            args.model,
            args.input,
            args.output,
            args.threshold,
            args.duration_threshold,
            args.keep_ratio,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 