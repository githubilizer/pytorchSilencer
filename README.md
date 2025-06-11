# PyTorch Silencer (LSTM Edition)

A PyTorch-based application for identifying and marking unnecessary silences in audio transcripts using LSTM networks.

## Overview

This application is designed to learn from well-formatted transcripts with appropriate silences, and then identify silences that should be cut in new transcripts. It provides a user-friendly dark mode interface for training the model and processing transcripts.

## Features

- **LSTM Neural Network**: Uses Long Short-Term Memory networks to analyze sequential patterns in transcript data
- **Contextual Understanding**: Considers the relationships between words, silences, and punctuation
- **Training Module**: Train the model on example transcripts with good silence patterns
- **Prediction**: Process new transcripts to identify silences that should be cut
- **Visualization**: Monitor training progress with loss graphs
- **Sleek Dark Mode UI**: Modern, eye-friendly interface

## How It Works

The application takes a sequence-based approach to analyze transcripts:

1. **Sequence Processing**: Instead of treating each silence independently, the LSTM model considers sequences of words and silences to understand patterns.
2. **Contextual Features**: Features include word length, duration, preceding and following words, punctuation, and position in the transcript.
3. **Bidirectional LSTM**: The model uses bidirectional LSTM cells to analyze context in both directions.
4. **Sequence Prediction**: Predictions are made on sequences rather than individual silences, capturing the relationships between consecutive words.

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

## LSTM Model Architecture

The model uses a bidirectional LSTM architecture:

- **Input**: Sequences of word and silence features
- **LSTM Layers**: Bidirectional LSTM with multiple layers
- **Hidden Size**: Configurable hidden layer size
- **Output**: Sequence of predictions for each silence (keep or cut)

This approach allows the model to understand the natural rhythm of speech and identify unnatural or excessive silences based on context. 