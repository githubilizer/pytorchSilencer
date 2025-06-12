#!/usr/bin/env python3
"""
Simple Silence Cutter - Marks silences exceeding a specified duration
"""

import sys
import os
import argparse
from data_processor import TranscriptProcessor, TranscriptData

def process_with_duration_threshold(transcript_path, output_path, threshold_seconds=1.5):
    """Process a transcript by marking silences longer than threshold_seconds"""
    print(f"Reading transcript from {transcript_path}")
    transcript = TranscriptProcessor.parse_transcript(transcript_path)
    
    # Generate cut markers based on silence duration
    cut_markers = []
    
    for i in range(len(transcript.entries) - 1):
        current = transcript.entries[i]
        next_entry = transcript.entries[i+1]
        
        # Calculate silence duration
        silence_duration = next_entry.start_time - current.end_time
        
        # Mark for cutting if silence exceeds threshold
        if silence_duration > threshold_seconds:
            cut_markers.append(True)
        else:
            cut_markers.append(False)
    
    typical = TranscriptProcessor.estimate_typical_silence(transcript)

    # Save processed transcript with predicted remaining silence
    print(f"Saving processed transcript to {output_path}")
    remaining = [typical if m else 0.0 for m in cut_markers]
    TranscriptProcessor.save_processed_transcript(transcript, output_path, cut_markers, remaining)
    
    # Print summary
    total_silences = sum(1 for i, entry in enumerate(transcript.entries[:-1]) if 
                         transcript.entries[i+1].start_time - entry.end_time > 0.01)
    cut_silences = sum(cut_markers)
    
    print(f"Total silences: {total_silences}")
    print(f"Silences marked for cutting: {cut_silences} ({cut_silences/max(1, total_silences)*100:.1f}%)")
    
    print("Done!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Simple silence cutter for transcripts")
    parser.add_argument("--input", "-i", required=True, help="Input transcript file")
    parser.add_argument("--output", "-o", help="Output transcript file (default: input_cleaned.txt)")
    parser.add_argument("--threshold", "-t", type=float, default=1.5, 
                       help="Silence duration threshold in seconds (default: 1.5)")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        base_name, ext = os.path.splitext(args.input)
        args.output = f"{base_name}_cleaned{ext}"
    
    process_with_duration_threshold(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main() 