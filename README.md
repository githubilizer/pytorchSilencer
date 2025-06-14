# PyTorch Silencer

A PyTorch-based application for identifying and marking unnecessary silences in audio transcripts. It includes both a supervised LSTM classifier and an autoencoder-based anomaly detector so you can train on good examples only.

## Overview

This application is designed to learn from well-formatted transcripts with appropriate silences, and then identify silences that should be cut in new transcripts. It provides a user-friendly dark mode interface for training the model and processing transcripts.

## Features

- **LSTM Neural Network**: Uses Long Short-Term Memory networks to analyze sequential patterns in transcript data
- **Autoencoder Anomaly Detection**: Learns from good examples only and flags unusual silences
- **Contextual Understanding**: Considers the relationships between words, silences, and punctuation
- **Training Module**: Train the model on example transcripts with good silence patterns
- **Prediction**: Process new transcripts to identify silences that should be cut
- **Visualization**: Monitor training progress with loss graphs
- **Sleek Dark Mode UI**: Modern, eye-friendly interface
- **Audio Silencer**: Trim video/audio based on processed transcripts with automatic crossfading and optional *middle-out* cuts when a remaining silence duration is provided

## How It Works

The application takes a sequence-based approach to analyze transcripts:

1. **Sequence Processing**: Instead of treating each silence independently, the LSTM model considers sequences of words and silences to understand patterns.
2. **Contextual Features**: Features include word length, duration, preceding and following words, punctuation, and position in the transcript.
3. **Bidirectional LSTM**: The model uses bidirectional LSTM cells to analyze context in both directions.
4. **Sequence Prediction**: Predictions are made on sequences rather than individual silences, capturing the relationships between consecutive words.
5. **Autoencoder Option**: Train on good data only and cut silences with high reconstruction error.

## Requirements

See `requirements.txt` for detailed dependencies.

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
   This installs all Python dependencies. For the video cutting feature, make
   sure [ffmpeg](https://ffmpeg.org/) is installed and available on your PATH.

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. **Training**:
   - Navigate to the "Training" tab
   - Select a directory containing good transcript examples
   - Configure LSTM parameters (sequence length, stride)
   - Set training parameters
   - Click "Train Model"
   - Save the trained model

3. **Prediction**:
   - Navigate to the "Prediction" tab
   - Load a trained model
   - Open a transcript file for processing
   - Set the threshold and click "Predict Silences"
   - Save the processed output

When running from the command line, the LSTM model automatically uses a
long-silence threshold derived from the training data. You can override
this by passing `--duration-threshold` with a custom number of seconds.
Use `--keep-ratio` to specify what fraction of each cut silence should remain
in the saved transcript.

## Transcript Format

The application expects transcripts in the following format:

```
[32.94 -> 33.22] five
[33.22 -> 33.26] 
[33.26 -> 34.06] APVs,
[34.06 -> 34.26] 
[34.26 -> 34.44] and
...
```

Where each line contains:
- Timestamps in square brackets: `[start_time -> end_time]`
- The spoken word or text after the timestamp
- Empty text after timestamps represents silence

## Output Format

The output adds `[SILENCE-CUT]` markers to indicate silences that should be removed:

```
[32.94 -> 33.22] five
[33.22 -> 33.26] 
[33.26 -> 34.06] APVs,
[34.06 -> 34.26] 
[34.26 -> 34.44] and
[34.44 -> 34.58] 
[34.58 -> 35.26] 29
[35.26 -> 35.34] 
[35.34 -> 35.98] artillery
[35.98 -> 37.42] [SILENCE-CUT]
...
```

When a duration is included, e.g. `[SILENCE-CUT 0.68s]`, the number
represents how much of that silence should remain **after** the edit.
The Audio Silencer removes the rest from the middle of the silence and
uses that remaining duration as the crossfade length, so in this example
only `0.68` seconds of audio/video are blended between the surrounding
segments.

## LSTM Model Architecture

The model uses a bidirectional LSTM architecture:

- **Input**: Sequences of word and silence features
- **LSTM Layers**: Bidirectional LSTM with multiple layers
- **Hidden Size**: Configurable hidden layer size
- **Output**: Sequence of predictions for each silence (keep or cut)

This approach allows the model to understand the natural rhythm of speech and identify unnatural or excessive silences based on context.

## Autoencoder Anomaly Detection

For setups where you only have good transcripts, you can train the autoencoder model.
It learns the typical timing between words and silences and flags any silence
that does not fit this pattern.

- **Input**: Individual feature vectors for each word/silence pair
- **Training**: Reconstruction loss on good examples only
- **Prediction**: Silences with high reconstruction error are marked for cutting
