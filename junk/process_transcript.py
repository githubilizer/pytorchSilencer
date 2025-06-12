#!/usr/bin/env python3
"""
PyTorch Silencer - Command Line Script for Transcript Processing
This script demonstrates how to use the PyTorch Silencer API programmatically.
"""

import os
import sys
import argparse
from data_processor import TranscriptProcessor
from silence_model import SilencePredictor

def process_transcript(model_path, input_file, output_file, threshold=0.5):
    """Process a transcript file using the trained model"""
    print(f"Loading model from {model_path}")
    predictor = SilencePredictor(model_path)
    
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
    
    typical = TranscriptProcessor.estimate_typical_silence(transcript)

    print(f"Saving processed transcript to {output_file}")
    remaining = [typical if m else 0.0 for m in cut_predictions]
    TranscriptProcessor.save_processed_transcript(transcript, output_file, cut_predictions, remaining)
    
    print("Done!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Process transcripts to mark silences for cutting")
    parser.add_argument("--model", "-m", required=True, help="Path to trained model file")
    parser.add_argument("--input", "-i", required=True, help="Input transcript file")
    parser.add_argument("--output", "-o", help="Output transcript file (default: input_cleaned.txt)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Prediction threshold (0-1)")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        base_name, ext = os.path.splitext(args.input)
        args.output = f"{base_name}_cleaned{ext}"
    
    # Process the transcript
    success = process_transcript(args.model, args.input, args.output, args.threshold)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 