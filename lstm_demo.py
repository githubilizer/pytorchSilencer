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
from silence_model import SilenceDurationPredictor, SilencePredictor
import time

def train_model(good_dir, model_path, sequence_length=10, epochs=50, batch_size=8, bad_dir=None):
    """Train a new silence prediction model.

    If ``bad_dir`` is provided, a classifier is trained using silences from
    ``good_dir`` as positive examples and those from ``bad_dir`` as negatives.
    Otherwise a duration model is trained using only ``good_dir``.
    """
    print(
        f"Training model with sequence length {sequence_length}, {epochs} epochs, batch size {batch_size}"
    )
    
    try:
        if bad_dir:
            features, labels = TranscriptProcessor.load_labeled_sequences(
                good_dir,
                bad_dir,
                sequence_length=sequence_length,
                stride=1,
            )
            good_feats, _ = TranscriptProcessor.load_duration_sequences(
                good_dir,
                sequence_length=sequence_length,
                stride=1,
            )
        else:
            features, labels = TranscriptProcessor.load_duration_sequences(
                good_dir,
                sequence_length=sequence_length,
                stride=1,
            )
            good_feats = features

        if len(features) == 0:
            print("No training data found. Please check the data directory.")
            return False

        print(f"Loaded {len(features)} training sequences with shape {features.shape}")

        # Derive long silence threshold from good training data
        silence_vals = good_feats[:, :, 2].reshape(-1)
        silence_vals = silence_vals[silence_vals > 0]
        long_thresh = float(np.percentile(silence_vals, 95)) if len(silence_vals) > 0 else None
        if long_thresh is not None:
            print(f"Derived long silence threshold: {long_thresh:.2f}s")

        if bad_dir:
            predictor = SilencePredictor(
                sequence_length=sequence_length,
                force_cpu=False,
                long_silence_threshold=long_thresh,
            )
        else:
            predictor = SilenceDurationPredictor(
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
        Unused. Maintained for backward compatibility.
    duration_threshold : float, optional
        Override for long silence detection. If None, the value
        stored in the model will be used.
    keep_ratio : float, optional
        (Deprecated) Previously indicated what fraction of a cut silence to
        keep. This value is ignored; the processor now outputs the exact
        extra silence duration to remove.
    """
    try:
        # Load the duration model
        predictor = SilenceDurationPredictor(model_path=model_path, force_cpu=False)
        
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
        
        # Predict ideal remaining silence durations
        print("Predicting target silence durations...")
        predicted = predictor.predict_durations(features)

        cut_markers = []
        cut_amounts = []
        for i, entry in enumerate(transcript.entries[:-1]):
            actual = transcript.entries[i + 1].start_time - entry.end_time
            desired = predicted[i] if i < len(predicted) else actual
            cut = max(actual - desired, 0.0)
            cut_markers.append(cut > 0.0)
            cut_amounts.append(cut)

        # Fallback: enforce cutting of very long silences
        effective_threshold = duration_threshold
        if effective_threshold is None:
            effective_threshold = predictor.long_silence_threshold

        if effective_threshold is not None:
            for i, entry in enumerate(transcript.entries[:-1]):
                silence_duration = transcript.entries[i+1].start_time - entry.end_time
                if silence_duration >= effective_threshold:
                    cut_markers[i] = True
                    desired = predicted[i] if i < len(predicted) else 0.0
                    cut_amounts[i] = max(silence_duration - desired, cut_amounts[i])
        
        # Calculate statistics using final cut markers
        total_silences = len(cut_markers)
        silences_to_cut = sum(1 for cut in cut_markers if cut)
        cut_percentage = silences_to_cut / total_silences * 100 if total_silences > 0 else 0

        print(f"Total silences: {total_silences}")
        print(f"Silences marked for cutting: {silences_to_cut} ({cut_percentage:.1f}%)")
        
        # Save processed transcript if output path provided
        if output_path:
            print(f"Saving processed transcript to {output_path}")
            TranscriptProcessor.save_processed_transcript(
                transcript,
                output_path,
                cut_markers,
                cut_amounts,
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
    train_parser.add_argument(
        "--data",
        "--good",
        dest="good",
        required=True,
        help="Directory with good transcript examples",
    )
    train_parser.add_argument(
        "--bad",
        help="Directory with bad transcript examples",
        default=None,
    )
    train_parser.add_argument("--model", required=True, help="Path to save model")
    train_parser.add_argument("--seq-length", type=int, default=10, help="Sequence length for LSTM")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    
    # Process subcommand
    process_parser = subparsers.add_parser("process", help="Process a transcript")
    process_parser.add_argument("--model", required=True, help="Path to trained model")
    process_parser.add_argument("--input", required=True, help="Input transcript file")
    process_parser.add_argument("--output", help="Output transcript file")
    process_parser.add_argument("--threshold", type=float, default=0.5, help="Ignored parameter for backward compatibility")
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
        train_model(
            args.good,
            args.model,
            sequence_length=args.seq_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            bad_dir=args.bad,
        )
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