# PyTorch Silencer - Usage Guide

This guide explains how to use the PyTorch Silencer application to process transcripts and mark unnecessary silences.

## Quick Start

The easiest way to run the application is to use the provided wrapper script:

```bash
# Run the GUI application (default)
./silencer.sh

# Process a transcript using the simple mode (duration-based)
./silencer.sh --mode simple --input your_transcript.txt --threshold 1.5

# Run the LSTM-based processor
./silencer.sh --mode cli --input your_transcript.txt --model models/lstm_model.pt
```

## Modes

The application has three modes:

1. **GUI Mode** (`--mode gui`): The graphical user interface with visualization and interactive controls
2. **CLI Mode** (`--mode cli`): Command-line LSTM-based processor with advanced ML features
3. **Simple Mode** (`--mode simple`): Fast, rule-based processor that marks silences exceeding a specified duration

## Common Parameters

These parameters work across all modes:

- `--input`, `-i`: Input transcript file path
- `--output`, `-o`: Output transcript file path
- `--threshold`, `-t`: In simple mode, this is the silence duration threshold in seconds (default: 1.5). In CLI mode, this is the prediction threshold (0-1, default: 0.5)

## LSTM Mode Parameters

When using CLI mode with LSTM processing:

- `--model`, `-m`: Path to LSTM model file (default: models/lstm_model.pt)
- `--data`, `-d`: Path to training data directory (default: sample_data)
- `--train`: Force model training even if model exists
- `--seq-length`: Sequence length for LSTM (default: 10)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Training batch size (default: 8)
- `--duration-threshold`: Override the model's long-silence threshold
  (in seconds). By default the threshold is learned from training data.
- `--keep-ratio`: Fraction of each cut silence to keep when saving the
  processed transcript (default: 0.5)

## Example Workflows

### Quick Silence Cutting

For immediate results without training, use the simple mode:

```bash
./silencer.sh --mode simple --input sample_data/long_silences.txt --threshold 2.0
```

This will mark all silences longer than 2 seconds for cutting.

### Training and Using an LSTM Model

1. Train a model on your good transcript examples:

```bash
./silencer.sh --mode cli --data your_training_data/ --model models/your_model.pt --train
```

2. Use the trained model to process a transcript:

```bash
./silencer.sh --mode cli --input your_transcript.txt --model models/your_model.pt
```

### Using the Autoencoder

If you only have good examples, train the autoencoder and cut anything that
doesn't match:

```bash
python autoencoder_demo.py train --data your_training_data --model models/ae_model.pt
python autoencoder_demo.py process --model models/ae_model.pt --input new_transcript.txt --output cleaned.txt
```

### Using the GUI

Just run the application without parameters to launch the GUI:

```bash
./silencer.sh
```

Then use the interface to:
1. Train a model (Training tab)
2. Process transcripts (Prediction tab)
3. Cut video using processed transcript (Audio Silencer tab)
