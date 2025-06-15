#!/usr/bin/env python3
"""
PyTorch Silencer - Command Line Script for Transcript Processing
This script demonstrates how to use the PyTorch Silencer API programmatically.
"""

import os
import sys
import argparse
from data_processor import TranscriptProcessor
from silence_model import SilenceDurationPredictor

def process_transcript(model_path, input_file, output_file, duration_threshold=None):
    """Process a transcript file using the duration model."""
    print(f"Loading model from {model_path}")
    predictor = SilenceDurationPredictor(model_path=model_path)
    
    print(f"Reading transcript from {input_file}")
    transcript = TranscriptProcessor.parse_transcript(input_file)
    
    print(f"Extracting features...")
    features = transcript.to_features()
    
    if len(features) == 0:
        print("Error: No valid features found in transcript.")
        return False
    
    print("Predicting desired pause lengths...")
    predicted = predictor.predict_durations(features)

    cut_markers = []
    cut_amounts = []
    for i, entry in enumerate(transcript.entries[:-1]):
        actual = transcript.entries[i + 1].start_time - entry.end_time
        desired = predicted[i] if i < len(predicted) else actual
        extra = max(actual - desired, 0.0)
        cut_markers.append(extra > 0.0)
        cut_amounts.append(extra)

    if duration_threshold is not None:
        for i, entry in enumerate(transcript.entries[:-1]):
            actual = transcript.entries[i + 1].start_time - entry.end_time
            if actual >= duration_threshold:
                desired = predicted[i] if i < len(predicted) else 0.0
                cut_markers[i] = True
                cut_amounts[i] = max(actual - desired, cut_amounts[i])

    print(f"Saving processed transcript to {output_file}")
    TranscriptProcessor.save_processed_transcript(
        transcript,
        output_file,
        cut_markers,
        cut_amounts,
    )
    
    print("Done!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process transcripts to mark silences for cutting")
    parser.add_argument("--model", "-m", required=True, help="Path to trained model file")
    parser.add_argument("--input", "-i", required=True, help="Input transcript file")
    parser.add_argument("--output", "-o", help="Output transcript file (default: input_cleaned.txt)")
    parser.add_argument(
        "--duration-threshold",
        "-t",
        type=float,
        default=None,
        help="Override long-silence threshold in seconds",
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        base_name, ext = os.path.splitext(args.input)
        args.output = f"{base_name}_cleaned{ext}"
    
    # Process the transcript
    success = process_transcript(
        args.model, args.input, args.output, args.duration_threshold
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 