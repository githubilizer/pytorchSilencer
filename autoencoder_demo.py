#!/usr/bin/env python3
"""Autoencoder-based silence detection demo."""
import argparse
import os
from data_processor import TranscriptProcessor
from anomaly_model import AutoencoderPredictor


def train_model(data_dir, model_path, epochs=50, batch_size=32):
    print(f"Training autoencoder with {epochs} epochs")
    features, _ = TranscriptProcessor.load_training_data(data_dir)
    if len(features) == 0:
        print("No training data found.")
        return False
    predictor = AutoencoderPredictor()
    predictor.train(features, epochs=epochs, batch_size=batch_size)
    predictor.save_model(model_path)
    print(f"Model saved to {model_path}")
    return True


def process_transcript(model_path, input_path, output_path=None, threshold=0.01):
    predictor = AutoencoderPredictor(model_path=model_path)
    transcript = TranscriptProcessor.parse_transcript(input_path)
    features = transcript.to_features()
    keep = predictor.predict(features, threshold=threshold)
    cut_markers = [not k for k in keep]
    if output_path:
        typical = TranscriptProcessor.estimate_typical_silence(transcript)
        remaining = [typical if m else 0.0 for m in cut_markers]
        TranscriptProcessor.save_processed_transcript(transcript, output_path, cut_markers, remaining)
    total = len(keep)
    cut = sum(cut_markers)
    print(f"Silences marked for cutting: {cut}/{total}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Autoencoder silence detector")
    subparsers = parser.add_subparsers(dest="cmd")

    t = subparsers.add_parser("train", help="Train a model")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=32)

    p = subparsers.add_parser("process", help="Process a transcript")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--threshold", type=float, default=0.01)

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(args.data, args.model, args.epochs, args.batch_size)
    elif args.cmd == "process":
        process_transcript(args.model, args.input, args.output, args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
