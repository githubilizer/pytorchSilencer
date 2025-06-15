import sys
import os
import subprocess
import datetime
import json
import re
import io
import contextlib
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QTextEdit, QTabWidget, 
                            QSpinBox, QDoubleSpinBox, QProgressBar,
                            QGroupBox, QMessageBox, QLineEdit, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont

class WorkerThread(QThread):
    """Run a Python callable in a background thread and capture its output."""

    progress_update = pyqtSignal(str)
    work_finished = pyqtSignal(bool, str)

    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                success = self.target(*self.args, **self.kwargs)
            for line in buffer.getvalue().splitlines():
                self.progress_update.emit(line)
            self.work_finished.emit(bool(success), "Completed")
        except Exception as e:
            self.work_finished.emit(False, str(e))


###########################################
# Data processing and ML utilities
###########################################

class TranscriptEntry:
    def __init__(self, start_time: float, end_time: float, text: str, silence_after: float = 0.0):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.silence_after = silence_after

    def __repr__(self):
        return f"[{self.start_time:.2f} -> {self.end_time:.2f}] {self.text}"


class TranscriptData:
    def __init__(self):
        self.entries: List[TranscriptEntry] = []

    def add_entry(self, entry: TranscriptEntry):
        self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def to_features(self) -> np.ndarray:
        features = []
        for i, entry in enumerate(self.entries[:-1]):
            if entry.text.strip() == "":
                continue
            silence_duration = self.entries[i + 1].start_time - entry.end_time
            prev_word_len = prev_word_duration = prev_silence_duration = 0
            if i > 0 and self.entries[i - 1].text.strip():
                prev_word_len = len(self.entries[i - 1].text)
                prev_word_duration = self.entries[i - 1].end_time - self.entries[i - 1].start_time
                prev_silence_duration = entry.start_time - self.entries[i - 1].end_time

            next_word_len = next_word_duration = 0
            if i + 1 < len(self.entries) and self.entries[i + 1].text.strip():
                next_word_len = len(self.entries[i + 1].text)
                next_word_duration = self.entries[i + 1].end_time - self.entries[i + 1].start_time

            entry_features = [
                len(entry.text),
                entry.end_time - entry.start_time,
                silence_duration,
                prev_word_len,
                prev_word_duration,
                prev_silence_duration,
                next_word_len,
                next_word_duration,
                1 if "," in entry.text else 0,
                1 if "." in entry.text else 0,
                1 if "?" in entry.text else 0,
                1 if "!" in entry.text else 0,
                entry.start_time,
                float(i) / len(self.entries),
            ]
            features.append(entry_features)

        return np.array(features) if features else np.array([])

    def get_contextual_sequences(self, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        features = self.to_features()
        if len(features) == 0:
            return np.array([]), np.array([])

        silences = []
        for i in range(len(self.entries) - 1):
            if self.entries[i].text.strip() == "":
                continue
            next_entry = self.entries[i + 1]
            silences.append(next_entry.start_time - self.entries[i].end_time)

        feature_sequences = []
        silence_sequences = []
        for i in range(0, len(features) - sequence_length + 1, stride):
            feature_seq = features[i:i + sequence_length]
            silence_seq = silences[i:i + sequence_length]
            if len(feature_seq) < sequence_length:
                pad_features = np.zeros((sequence_length - len(feature_seq), features.shape[1]))
                feature_seq = np.vstack([feature_seq, pad_features])
                pad_silences = np.zeros(sequence_length - len(silence_seq))
                silence_seq = np.concatenate([silence_seq, pad_silences])
            feature_sequences.append(feature_seq)
            silence_sequences.append(silence_seq)

        return np.array(feature_sequences), np.array(silence_sequences)


class TranscriptProcessor:
    @staticmethod
    def parse_transcript(file_path: str) -> TranscriptData:
        transcript = TranscriptData()
        with open(file_path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            timestamp_match = re.match(r"\[([\d\.]+) -> ([\d\.]+)\] (.*)", line)
            if timestamp_match:
                start_time = float(timestamp_match.group(1))
                end_time = float(timestamp_match.group(2))
                text = timestamp_match.group(3)
                transcript.add_entry(TranscriptEntry(start_time, end_time, text))
            i += 1

        for i in range(len(transcript.entries) - 1):
            current = transcript.entries[i]
            next_entry = transcript.entries[i + 1]
            if next_entry.text.strip():
                current.silence_after = next_entry.start_time - current.end_time
        return transcript

    @staticmethod
    def estimate_typical_silence(transcript: TranscriptData, max_duration: float = 1.0) -> float:
        silences = []
        for i in range(len(transcript.entries) - 1):
            a = transcript.entries[i]
            b = transcript.entries[i + 1]
            if a.text.strip() and b.text.strip():
                dur = b.start_time - a.end_time
                if 0 < dur < max_duration:
                    silences.append(dur)
        if silences:
            return float(np.median(silences))
        return 0.1

    @staticmethod
    def save_processed_transcript(
        transcript: TranscriptData,
        output_path: str,
        cut_markers: List[bool],
        cut_durations: Optional[List[float]] = None,
    ):
        with open(output_path, "w") as f:
            for i, entry in enumerate(transcript.entries):
                f.write(f"[{entry.start_time:.2f} -> {entry.end_time:.2f}] {entry.text}\n")

                if i < len(transcript.entries) - 1:
                    next_entry = transcript.entries[i + 1]
                    silence_duration = next_entry.start_time - entry.end_time
                    if silence_duration > 0.01:
                        if i < len(cut_markers) and cut_markers[i]:
                            cut = None
                            if cut_durations and i < len(cut_durations):
                                cut = cut_durations[i]
                            if cut is not None:
                                f.write(
                                    f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] [SILENCE-CUT {cut:.2f}s]\n"
                                )
                            else:
                                f.write(
                                    f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] [SILENCE-CUT]\n"
                                )
                        else:
                            f.write(f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] \n")

        print(f"Processed transcript saved to {output_path}")

    @staticmethod
    def load_duration_sequences(directory: str, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        all_feature_sequences = []
        all_duration_sequences = []
        for filename in os.listdir(directory):
            if not filename.endswith(".txt"):
                continue
            file_path = os.path.join(directory, filename)
            transcript = TranscriptProcessor.parse_transcript(file_path)
            feat_seqs, dur_seqs = transcript.get_contextual_sequences(sequence_length, stride)
            if len(feat_seqs) > 0:
                all_feature_sequences.append(feat_seqs)
                all_duration_sequences.append(dur_seqs)
        if all_feature_sequences:
            return np.vstack(all_feature_sequences), np.vstack(all_duration_sequences)
        return np.array([]), np.array([])

    @staticmethod
    def load_labeled_sequences(good_directory: str, bad_directory: str, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        feature_sequences = []
        label_sequences = []

        def add_from_dir(directory: str, label: int):
            if not directory:
                return
            for fname in os.listdir(directory):
                if not fname.endswith(".txt"):
                    continue
                path = os.path.join(directory, fname)
                transcript = TranscriptProcessor.parse_transcript(path)
                feats, _ = transcript.get_contextual_sequences(sequence_length, stride)
                if len(feats) > 0:
                    feature_sequences.append(feats)
                    label_sequences.append(np.full((feats.shape[0], feats.shape[1]), label, dtype=np.float32))

        add_from_dir(good_directory, 1)
        add_from_dir(bad_directory, 0)

        if feature_sequences:
            return np.vstack(feature_sequences), np.vstack(label_sequences)
        return np.array([]), np.array([])


class LSTMSilenceClassifier(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        batch_size, seq_len, _ = lstm_out.size()
        fc_in = lstm_out.contiguous().view(batch_size * seq_len, self.hidden_size * 2)
        fc_out = self.fc(fc_in)
        outputs = fc_out.view(batch_size, seq_len, 1)
        return outputs


class LSTMSilenceRegressor(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        batch_size, seq_len, _ = lstm_out.size()
        fc_in = lstm_out.contiguous().view(batch_size * seq_len, self.hidden_size * 2)
        fc_out = self.fc(fc_in)
        outputs = fc_out.view(batch_size, seq_len, 1)
        return outputs


class SequenceScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        if len(X.shape) == 2:
            X_reshaped = X
        else:
            X_reshaped = X.reshape(-1, X.shape[2])
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet.")
        original_shape = X.shape
        is_2d = len(original_shape) == 2
        if is_2d:
            X_reshaped = X
        else:
            X_reshaped = X.reshape(-1, original_shape[2])
        X_scaled = self.scaler.transform(X_reshaped)
        if not is_2d:
            X_scaled = X_scaled.reshape(original_shape)
        return X_scaled


def get_device_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if info["cuda_available"]:
        for i in range(info["device_count"]):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
            }
            info["devices"].append(device_info)
    return info


class SilencePredictor:
    def __init__(self, model_path: Optional[str] = None, sequence_length: int = 10, input_size: int = 14, force_cpu: bool = False, long_silence_threshold: Optional[float] = None):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.cuda_info = get_device_info()
        if not force_cpu and self.cuda_info["cuda_available"]:
            best_device_idx = 0
            best_capability = (0, 0)
            for i, device in enumerate(self.cuda_info["devices"]):
                if device["capability"] > best_capability:
                    best_capability = device["capability"]
                    best_device_idx = i
            self.device = torch.device(f"cuda:{best_device_idx}")
        else:
            self.device = torch.device("cpu")

        self.model = LSTMSilenceClassifier(self.input_size)
        self.scaler = SequenceScaler()
        self.long_silence_threshold = long_silence_threshold
        self.model.to(self.device)
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, features: np.ndarray, labels: np.ndarray, epochs: int = 100, batch_size: int = 8, learning_rate: float = 0.001) -> List[float]:
        self.scaler.fit(features)
        X = torch.FloatTensor(self.scaler.transform(features)).to(self.device)
        y = torch.FloatTensor(labels).to(self.device).unsqueeze(-1)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        losses = []
        for epoch in range(epochs):
            indices = torch.randperm(X.size(0))
            epoch_loss = 0.0
            num_batches = 0
            for start_idx in range(0, X.size(0), batch_size):
                end_idx = min(start_idx + batch_size, X.size(0))
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            losses.append(epoch_loss / num_batches)
        return losses

    def predict_scores(self, features: np.ndarray) -> List[float]:
        if len(features) == 0:
            return []
        seq_length = self.sequence_length
        n_features = features.shape[1]
        if len(features) < seq_length:
            padding = np.zeros((seq_length - len(features), n_features))
            features_padded = np.vstack([features, padding])
            sequences = np.array([features_padded])
        else:
            n_sequences = len(features) - seq_length + 1
            sequences = np.array([features[i:i + seq_length] for i in range(n_sequences)])
        scaled_sequences = self.scaler.transform(sequences)
        X = torch.FloatTensor(scaled_sequences).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X).squeeze(-1).cpu().numpy()
        scores = []
        if len(features) <= seq_length:
            scores = outputs[0, :len(features)]
        else:
            accumulated = np.zeros(len(features))
            counts = np.zeros(len(features))
            for i in range(len(outputs)):
                for j in range(seq_length):
                    idx = i + j
                    if idx < len(features):
                        accumulated[idx] += outputs[i, j]
                        counts[idx] += 1
            for i in range(len(features)):
                if counts[i] > 0:
                    scores.append(accumulated[i] / counts[i])
                else:
                    scores.append(1.0)
        return [float(p) for p in scores]

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> List[bool]:
        scores = self.predict_scores(features)
        return [s >= threshold for s in scores]

    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "long_silence_threshold": self.long_silence_threshold,
        }, model_path)
        with open(model_path + ".scaler", "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.sequence_length = checkpoint.get("sequence_length", 10)
        self.input_size = checkpoint.get("input_size", self.input_size)
        self.long_silence_threshold = checkpoint.get("long_silence_threshold")
        self.model = LSTMSilenceClassifier(self.input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        scaler_path = model_path + ".scaler"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)


class SilenceDurationPredictor:
    def __init__(self, model_path: Optional[str] = None, sequence_length: int = 10, input_size: int = 14, force_cpu: bool = False, long_silence_threshold: Optional[float] = None):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.cuda_info = get_device_info()
        if not force_cpu and self.cuda_info["cuda_available"]:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model = LSTMSilenceRegressor(self.input_size).to(self.device)
        self.scaler = SequenceScaler()
        self.long_silence_threshold = long_silence_threshold
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, features: np.ndarray, durations: np.ndarray, epochs: int = 100, batch_size: int = 8, learning_rate: float = 0.001) -> List[float]:
        self.scaler.fit(features)
        X = torch.FloatTensor(self.scaler.transform(features)).to(self.device)
        y = torch.FloatTensor(durations).to(self.device).unsqueeze(-1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        losses = []
        for _ in range(epochs):
            indices = torch.randperm(X.size(0))
            epoch_loss = 0.0
            num_batches = 0
            for start_idx in range(0, X.size(0), batch_size):
                end_idx = min(start_idx + batch_size, X.size(0))
                batch_idx = indices[start_idx:end_idx]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            losses.append(epoch_loss / num_batches)
        return losses

    def predict_durations(self, features: np.ndarray) -> List[float]:
        if len(features) == 0:
            return []
        seq_length = self.sequence_length
        n_features = features.shape[1]
        if len(features) < seq_length:
            padding = np.zeros((seq_length - len(features), n_features))
            features_padded = np.vstack([features, padding])
            sequences = np.array([features_padded])
        else:
            n_sequences = len(features) - seq_length + 1
            sequences = np.array([features[i:i + seq_length] for i in range(n_sequences)])
        scaled_sequences = self.scaler.transform(sequences)
        X = torch.FloatTensor(scaled_sequences).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X).squeeze(-1).cpu().numpy()
        preds = []
        if len(features) <= seq_length:
            preds = outputs[0, :len(features)].tolist()
        else:
            accumulated = np.zeros(len(features))
            counts = np.zeros(len(features))
            for i in range(len(outputs)):
                for j in range(seq_length):
                    idx = i + j
                    if idx < len(features):
                        accumulated[idx] += outputs[i, j]
                        counts[idx] += 1
            for i in range(len(features)):
                if counts[i] > 0:
                    preds.append(accumulated[i] / counts[i])
                else:
                    preds.append(0.0)
        return [float(p) for p in preds]

    def save_model(self, model_path: str):
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "long_silence_threshold": self.long_silence_threshold,
            "duration_model": True,
        }, model_path)
        with open(model_path + ".scaler", "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.sequence_length = checkpoint.get("sequence_length", 10)
        self.input_size = checkpoint.get("input_size", self.input_size)
        self.long_silence_threshold = checkpoint.get("long_silence_threshold")
        self.model = LSTMSilenceRegressor(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        scaler_path = model_path + ".scaler"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)


def train_model(good_dir: str, model_path: str, sequence_length: int = 10, epochs: int = 50, batch_size: int = 8, bad_dir: Optional[str] = None) -> bool:
    print(f"Training model with sequence length {sequence_length}, {epochs} epochs, batch size {batch_size}")
    try:
        if bad_dir:
            features, labels = TranscriptProcessor.load_labeled_sequences(good_dir, bad_dir, sequence_length=sequence_length, stride=1)
            good_feats, _ = TranscriptProcessor.load_duration_sequences(good_dir, sequence_length=sequence_length, stride=1)
        else:
            features, labels = TranscriptProcessor.load_duration_sequences(good_dir, sequence_length=sequence_length, stride=1)
            good_feats = features
        if len(features) == 0:
            print("No training data found. Please check the data directory.")
            return False
        silence_vals = good_feats[:, :, 2].reshape(-1)
        silence_vals = silence_vals[silence_vals > 0]
        long_thresh = float(np.percentile(silence_vals, 95)) if len(silence_vals) > 0 else None
        if bad_dir:
            predictor = SilencePredictor(sequence_length=sequence_length, force_cpu=False, long_silence_threshold=long_thresh)
        else:
            predictor = SilenceDurationPredictor(sequence_length=sequence_length, force_cpu=False, long_silence_threshold=long_thresh)
        for epoch in range(1, epochs + 1):
            print(f"Starting Epoch {epoch}/{epochs}...")
            predictor.train(features, labels, epochs=1, batch_size=batch_size)
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"Progress: {int(epoch / epochs * 100)}% complete")
        predictor.save_model(model_path)
        print(f"Model saved to {model_path}")
        return True
    except Exception as e:
        import traceback
        print(f"Training error: {str(e)}")
        print(traceback.format_exc())
        return False


def process_transcript(model_path: str, input_path: str, output_path: Optional[str] = None, threshold: float = 0.5, duration_threshold: Optional[float] = None, keep_ratio: float = 0.5) -> bool:
    try:
        predictor = SilenceDurationPredictor(model_path=model_path, force_cpu=False)
        print(f"Loading transcript from {input_path}")
        transcript = TranscriptProcessor.parse_transcript(input_path)
        if len(transcript.entries) == 0:
            print("No entries found in transcript.")
            return False
        features = transcript.to_features()
        if len(features) == 0:
            print("No features extracted from transcript.")
            return False
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
        effective_threshold = duration_threshold
        if effective_threshold is None:
            effective_threshold = predictor.long_silence_threshold
        if effective_threshold is not None:
            for i, entry in enumerate(transcript.entries[:-1]):
                silence_duration = transcript.entries[i + 1].start_time - entry.end_time
                if silence_duration >= effective_threshold:
                    cut_markers[i] = True
                    desired = predicted[i] if i < len(predicted) else 0.0
                    cut_amounts[i] = max(silence_duration - desired, cut_amounts[i])
        total_silences = len(cut_markers)
        silences_to_cut = sum(1 for cut in cut_markers if cut)
        cut_percentage = silences_to_cut / total_silences * 100 if total_silences > 0 else 0
        print(f"Total silences: {total_silences}")
        print(f"Silences marked for cutting: {silences_to_cut} ({cut_percentage:.1f}%)")
        if output_path:
            print(f"Saving processed transcript to {output_path}")
            TranscriptProcessor.save_processed_transcript(transcript, output_path, cut_markers, cut_amounts)
        return True
    except Exception as e:
        import traceback
        print(f"Processing error: {str(e)}")
        print(traceback.format_exc())
        return False


def parse_cut_segments(path: str) -> List[Tuple[float, float, Optional[float]]]:
    segments: List[Tuple[float, float, Optional[float]]] = []
    timestamp = re.compile(r"\[(\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)\]")
    cut_re = re.compile(r"\[SILENCE-CUT(?: ([\d\.]+)s)?\]")
    with open(path, "r") as f:
        for line in f:
            if "[SILENCE-CUT" not in line:
                continue
            m = timestamp.search(line)
            if not m:
                continue
            cut_m = cut_re.search(line)
            remain = float(cut_m.group(1)) if cut_m and cut_m.group(1) else None
            start = float(m.group(1))
            end = float(m.group(2))
            segments.append((start, end, remain))
    return segments


def get_video_duration(path: str) -> float:
    result = subprocess.run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())


def get_video_fps(path: str) -> float:
    result = subprocess.run([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, denom = fps_str.split("/")
        return float(num) / float(denom)
    return float(fps_str)


def cut_video(video_path: str, transcript_path: str, output_path: str) -> None:
    segments = parse_cut_segments(transcript_path)
    video_duration = get_video_duration(video_path)
    fps = int(round(get_video_fps(video_path)))
    if not segments:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            output_path,
        ], check=True)
        return
    keep_segments: List[Tuple[float, float]] = []
    crossfades: List[float] = []
    current = 0.0
    for start, end, cut in segments:
        seg_dur = end - start
        if cut is not None and cut < seg_dur:
            keep_start = start + cut / 2
            keep_end = end - cut / 2
            remain = seg_dur - cut
            cf = max(remain, 0.0)
        else:
            keep_start = start + 0.1 * seg_dur
            keep_end = end - 0.1 * seg_dur
            cf = min(seg_dur * 0.1, 1.0)
        if keep_start > current:
            keep_segments.append((current, keep_start))
            crossfades.append(cf)
        current = keep_end
    if current < video_duration:
        keep_segments.append((current, video_duration))
    filter_parts = []
    for idx, (s, e) in enumerate(keep_segments):
        filter_parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,fps={fps}[v{idx}]")
        filter_parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{idx}]")
    vf = "v0"
    af = "a0"
    out_dur = keep_segments[0][1] - keep_segments[0][0]
    for idx in range(1, len(keep_segments)):
        cf = crossfades[idx - 1] if idx - 1 < len(crossfades) else 0
        offset = max(out_dur - cf, 0)
        filter_parts.append(f"[{vf}][v{idx}]xfade=transition=fade:duration={cf}:offset={offset}[vx{idx}]")
        filter_parts.append(f"[{af}][a{idx}]acrossfade=d={cf}[ax{idx}]")
        vf = f"vx{idx}"
        af = f"ax{idx}"
        out_dur = out_dur - cf + (keep_segments[idx][1] - keep_segments[idx][0])
    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{vf}]",
        "-map",
        f"[{af}]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True)


class NotesTextEdit(QTextEdit):
    """Simple text editor with autosave and a custom context menu."""

    def __init__(self, notes_file, parent=None):
        super().__init__(parent)
        self.notes_file = notes_file
        self.textChanged.connect(self.auto_save)
        self.load_notes()

    def load_notes(self):
        if os.path.exists(self.notes_file):
            try:
                with open(self.notes_file, "r", encoding="utf-8") as f:
                    self.setPlainText(f.read())
            except Exception:
                pass

    def auto_save(self):
        try:
            with open(self.notes_file, "w", encoding="utf-8") as f:
                f.write(self.toPlainText())
        except Exception:
            pass

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        select_all_copy = QAction("Select All && Copy", self)
        select_all_copy.triggered.connect(self.select_all_and_copy)
        menu.addSeparator()
        menu.addAction(select_all_copy)
        menu.exec_(event.globalPos())

    def select_all_and_copy(self):
        self.selectAll()
        self.copy()

class PytorchSilencerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch Silencer - GPU Edition")
        self.setGeometry(100, 100, 1200, 800)

        # Set dark mode
        self.setup_dark_theme()

        # Path persistence file
        self.PATHS_FILE = os.path.join(os.path.dirname(__file__), "last_paths.json")

        # Initialize paths
        self.good_data_dir = ""
        self.bad_data_dir = ""

        # Load last used model if available
        self.model_path = ""
        self.LAST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "last_model_path.txt")
        if os.path.exists(self.LAST_MODEL_FILE):
            try:
                with open(self.LAST_MODEL_FILE, "r") as f:
                    self.model_path = f.read().strip()
            except Exception:
                self.model_path = ""

        if not self.model_path:
            self.model_path = "/home/j/Desktop/code/pytorchSilencer/models/silence_model.pt"

        # Default input transcript path for the prediction tab
        self.input_transcript = ""
        self.output_transcript = ""

        # Default paths for Audio Silencer tab
        self.processed_transcript_path = ""
        self.input_video_path = ""
        self.output_video_path = ""

        # Load persisted paths if available
        self.load_paths()

        # If we loaded an input transcript but no processed output use timestamp
        if self.input_transcript and not self.output_transcript:
            base, ext = os.path.splitext(self.input_transcript)
            hhmm = datetime.datetime.now().strftime("%I%M%p").lower()
            self.output_transcript = f"{base}_processed_{hhmm}{ext}"

        if not self.processed_transcript_path:
            self.processed_transcript_path = self.output_transcript

        # Notes file path and editor placeholder
        self.NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.txt")
        self.notes_editor = None

        # Setup UI
        self.setup_ui()
        
    def setup_dark_theme(self):
        # Set dark palette
        self.dark_palette = QPalette()
        self.dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        self.dark_palette.setColor(QPalette.WindowText, Qt.white)
        self.dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        self.dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        self.dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        self.dark_palette.setColor(QPalette.Text, Qt.white)
        self.dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        self.dark_palette.setColor(QPalette.ButtonText, Qt.white)
        self.dark_palette.setColor(QPalette.BrightText, Qt.red)
        self.dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        self.dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        # Apply palette
        QApplication.setPalette(self.dark_palette)
        
    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Create training tab
        training_tab = QWidget()
        self.setup_training_tab(training_tab)
        self.tabs.addTab(training_tab, "Train Model")
        
        # Create prediction tab
        prediction_tab = QWidget()
        self.setup_prediction_tab(prediction_tab)
        self.tabs.addTab(prediction_tab, "Process Transcript")

        # Create audio silencer tab
        audio_tab = QWidget()
        self.setup_audio_tab(audio_tab)
        self.tabs.addTab(audio_tab, "Audio Silencer")

        # Create notes tab (always last)
        notes_tab = QWidget()
        self.setup_notes_tab(notes_tab)
        self.tabs.addTab(notes_tab, "Notes")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
    def setup_training_tab(self, tab):
        # Layout
        layout = QVBoxLayout(tab)
        
        # Training data selection
        data_group = QGroupBox("Training Data")
        data_layout = QVBoxLayout(data_group)

        good_layout = QHBoxLayout()
        self.good_data_path = QLineEdit()
        self.good_data_path.setReadOnly(True)
        self.good_data_path.setPlaceholderText("Select GOOD transcripts directory...")

        if self.good_data_dir:
            self.good_data_path.setText(self.good_data_dir)

        browse_good_button = QPushButton("Browse...")
        browse_good_button.clicked.connect(self.browse_good_data)

        # Add quick load button for GOODFULLSCRIPTSDATA
        quick_load_button = QPushButton("Load Full Scripts")
        quick_load_button.setToolTip("Load training data from GOODFULLSCRIPTSDATA folder")
        quick_load_button.clicked.connect(self.load_full_scripts_data)

        good_layout.addWidget(self.good_data_path)
        good_layout.addWidget(browse_good_button)
        good_layout.addWidget(quick_load_button)

        bad_layout = QHBoxLayout()
        self.bad_data_path = QLineEdit()
        self.bad_data_path.setReadOnly(True)
        self.bad_data_path.setPlaceholderText("Select BAD transcripts directory...")

        if self.bad_data_dir:
            self.bad_data_path.setText(self.bad_data_dir)

        browse_bad_button = QPushButton("Browse...")
        browse_bad_button.clicked.connect(self.browse_bad_data)

        bad_layout.addWidget(self.bad_data_path)
        bad_layout.addWidget(browse_bad_button)

        data_layout.addLayout(good_layout)
        data_layout.addLayout(bad_layout)
        
        # Model parameters
        param_group = QGroupBox("Model Parameters")
        param_layout = QHBoxLayout(param_group)
        
        # Sequence length
        seq_length_label = QLabel("Sequence Length:")
        self.seq_length_spinner = QSpinBox()
        self.seq_length_spinner.setRange(1, 50)
        self.seq_length_spinner.setValue(10)
        
        # Epochs
        epochs_label = QLabel("Epochs:")
        self.epochs_spinner = QSpinBox()
        self.epochs_spinner.setRange(1, 1000)
        self.epochs_spinner.setValue(50)
        
        # Batch size
        batch_size_label = QLabel("Batch Size:")
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 64)
        self.batch_size_spinner.setValue(8)
        
        # Add to parameter layout
        param_layout.addWidget(seq_length_label)
        param_layout.addWidget(self.seq_length_spinner)
        param_layout.addWidget(epochs_label)
        param_layout.addWidget(self.epochs_spinner)
        param_layout.addWidget(batch_size_label)
        param_layout.addWidget(self.batch_size_spinner)
        
        # Model path
        model_group = QGroupBox("Model Path")
        model_layout = QHBoxLayout(model_group)
        
        self.model_path_edit = QLineEdit(self.model_path)
        save_model_button = QPushButton("Save As...")
        save_model_button.clicked.connect(self.set_model_path)
        
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(save_model_button)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        
        button_layout.addStretch()
        button_layout.addWidget(self.train_btn)
        
        # Progress bar
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # Status text
        self.data_status_label = QLabel("No data loaded")
        
        # Log viewer
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        
        log_layout.addWidget(self.log_viewer)
        
        # Add widgets to progress layout
        progress_layout.addWidget(self.data_status_label)
        progress_layout.addWidget(self.progress_bar)

        # Add all components to main layout
        layout.addWidget(data_group)
        layout.addWidget(param_group)
        layout.addWidget(model_group)
        layout.addLayout(button_layout)
        layout.addWidget(progress_group)
        layout.addWidget(log_group, 1)  # Give log more stretch

        # Update data status now that label exists
        self.update_data_status()
        
    def setup_prediction_tab(self, tab):
        # Layout
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)
        
        self.model_path_field = QLineEdit()
        self.model_path_field.setReadOnly(False)
        self.model_path_field.setPlaceholderText("Select model file...")
        # Populate default model path
        if self.model_path:
            self.model_path_field.setText(self.model_path)
        
        browse_model_button = QPushButton("Browse...")
        browse_model_button.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path_field)
        model_layout.addWidget(browse_model_button)
        
        # Input transcript
        input_group = QGroupBox("Input Transcript")
        input_layout = QHBoxLayout(input_group)
        
        self.input_transcript_field = QLineEdit()
        self.input_transcript_field.setReadOnly(False)
        self.input_transcript_field.setPlaceholderText("Select input transcript file...")
        # Populate default input transcript path
        if self.input_transcript:
            self.input_transcript_field.setText(self.input_transcript)
        
        browse_input_button = QPushButton("Browse...")
        browse_input_button.clicked.connect(self.browse_input_transcript)
        
        input_layout.addWidget(self.input_transcript_field)
        input_layout.addWidget(browse_input_button)
        
        # Output transcript
        output_group = QGroupBox("Output Transcript")
        output_layout = QHBoxLayout(output_group)
        
        self.output_transcript_field = QLineEdit()
        self.output_transcript_field.setReadOnly(False)
        self.output_transcript_field.setPlaceholderText("Select output transcript file...")
        # Populate default output transcript path
        if self.output_transcript:
            self.output_transcript_field.setText(self.output_transcript)
        
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self.browse_output_transcript)
        
        output_layout.addWidget(self.output_transcript_field)
        output_layout.addWidget(browse_output_button)
        
        # Threshold
        threshold_group = QGroupBox("Prediction Threshold")
        threshold_layout = QHBoxLayout(threshold_group)
        
        threshold_label = QLabel("Threshold:")
        self.threshold_spinner = QDoubleSpinBox()
        self.threshold_spinner.setRange(0.0, 1.0)
        self.threshold_spinner.setSingleStep(0.05)
        self.threshold_spinner.setValue(0.5)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spinner)
        threshold_layout.addStretch()

        # Keep ratio
        keep_group = QGroupBox("Keep Ratio")
        keep_layout = QHBoxLayout(keep_group)

        keep_label = QLabel("Keep Ratio:")
        self.keep_ratio_spinner = QDoubleSpinBox()
        self.keep_ratio_spinner.setRange(0.0, 1.0)
        self.keep_ratio_spinner.setSingleStep(0.05)
        self.keep_ratio_spinner.setValue(0.5)

        keep_layout.addWidget(keep_label)
        keep_layout.addWidget(self.keep_ratio_spinner)
        keep_layout.addStretch()
        
        # Process and open buttons
        button_layout = QHBoxLayout()

        self.process_btn = QPushButton("Process Transcript")
        self.process_btn.clicked.connect(self.process_transcript)

        self.open_last_output_btn = QPushButton("Open Last Output Transcript")
        self.open_last_output_btn.clicked.connect(self.open_last_output_transcript)

        button_layout.addStretch()
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.open_last_output_btn)
        
        # Log viewer
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.process_log_viewer = QTextEdit()
        self.process_log_viewer.setReadOnly(True)
        
        log_layout.addWidget(self.process_log_viewer)

        # Add all components to main layout
        layout.addWidget(model_group)
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addWidget(threshold_group)
        layout.addWidget(keep_group)
        layout.addLayout(button_layout)
        layout.addWidget(log_group, 1)  # Give log more stretch

    def setup_audio_tab(self, tab):
        layout = QVBoxLayout(tab)

        # Processed transcript selection
        transcript_group = QGroupBox("Processed Transcript")
        transcript_layout = QHBoxLayout(transcript_group)

        self.audio_transcript_field = QLineEdit()
        self.audio_transcript_field.setReadOnly(False)
        self.audio_transcript_field.setPlaceholderText("Select processed transcript file...")
        # Populate default processed transcript path
        if self.processed_transcript_path:
            self.audio_transcript_field.setText(self.processed_transcript_path)

        browse_transcript_button = QPushButton("Browse...")
        browse_transcript_button.clicked.connect(self.browse_audio_transcript)

        transcript_layout.addWidget(self.audio_transcript_field)
        transcript_layout.addWidget(browse_transcript_button)

        # Video selection
        video_group = QGroupBox("Input Video")
        video_layout = QHBoxLayout(video_group)

        self.video_path_field = QLineEdit()
        self.video_path_field.setReadOnly(False)
        self.video_path_field.setPlaceholderText("Select video file...")
        if self.input_video_path:
            self.video_path_field.setText(self.input_video_path)

        browse_video_button = QPushButton("Browse...")
        browse_video_button.clicked.connect(self.browse_video_file)

        video_layout.addWidget(self.video_path_field)
        video_layout.addWidget(browse_video_button)

        # Output video
        output_video_group = QGroupBox("Output Video")
        output_video_layout = QHBoxLayout(output_video_group)

        self.output_video_field = QLineEdit()
        self.output_video_field.setReadOnly(False)
        self.output_video_field.setPlaceholderText("Select output video file...")
        if self.output_video_path:
            self.output_video_field.setText(self.output_video_path)

        browse_output_video_button = QPushButton("Browse...")
        browse_output_video_button.clicked.connect(self.browse_output_video)

        output_video_layout.addWidget(self.output_video_field)
        output_video_layout.addWidget(browse_output_video_button)

        # Process button
        button_layout = QHBoxLayout()

        self.audio_process_btn = QPushButton("Cut Video")
        self.audio_process_btn.clicked.connect(self.process_video)

        button_layout.addStretch()
        button_layout.addWidget(self.audio_process_btn)

        # Log viewer
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.audio_log_viewer = QTextEdit()
        self.audio_log_viewer.setReadOnly(True)

        log_layout.addWidget(self.audio_log_viewer)

        # Add widgets to layout
        layout.addWidget(transcript_group)
        layout.addWidget(video_group)
        layout.addWidget(output_video_group)
        layout.addLayout(button_layout)
        layout.addWidget(log_group, 1)

    def setup_notes_tab(self, tab):
        layout = QVBoxLayout(tab)
        self.notes_editor = NotesTextEdit(self.NOTES_FILE)
        layout.addWidget(self.notes_editor)
        
    def browse_good_data(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Good Transcript Directory")
        if directory:
            self.good_data_dir = directory
            self.good_data_path.setText(directory)
            self.update_data_status()
            self.save_paths()

    def browse_bad_data(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Bad Transcript Directory")
        if directory:
            self.bad_data_dir = directory
            self.bad_data_path.setText(directory)
            self.update_data_status()
            self.save_paths()

    def update_data_status(self):
        good_count = 0
        bad_count = 0
        if self.good_data_dir and os.path.isdir(self.good_data_dir):
            good_count = sum(1 for f in os.listdir(self.good_data_dir) if f.endswith('.txt'))
        if self.bad_data_dir and os.path.isdir(self.bad_data_dir):
            bad_count = sum(1 for f in os.listdir(self.bad_data_dir) if f.endswith('.txt'))
        self.data_status_label.setText(f"Good: {good_count} files   Bad: {bad_count} files")
            
    def set_model_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Model Path", self.model_path, "PyTorch Model (*.pt)"
        )
        if file_path:
            base, ext = os.path.splitext(file_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{base}_{timestamp}{ext}"
            self.model_path = file_path
            self.model_path_edit.setText(file_path)
            self.save_paths()
            
    def train_model(self):
        # Validate inputs
        if not self.good_data_dir or not self.bad_data_dir:
            QMessageBox.warning(self, "No Data", "Please select directories for both GOOD and BAD transcripts first.")
            return
            
        # Get parameters
        seq_length = self.seq_length_spinner.value()
        epochs = self.epochs_spinner.value()
        batch_size = self.batch_size_spinner.value()
        model_path = self.model_path_edit.text()
        base, ext = os.path.splitext(model_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{base}_{timestamp}{ext}"
        self.model_path_edit.setText(model_path)
        
        # Clear log
        self.log_viewer.clear()
        self.add_log_message("Starting training process...")
        
        # Disable buttons
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Run training in background thread
        self.command_thread = WorkerThread(
            train_model,
            self.good_data_dir,
            model_path,
            sequence_length=seq_length,
            epochs=epochs,
            batch_size=batch_size,
            bad_dir=self.bad_data_dir,
        )
        self.command_thread.progress_update.connect(self.add_log_message)
        self.command_thread.work_finished.connect(self.on_training_finished)
        self.command_thread.start()
            
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "models", "PyTorch Model (*.pt)"
        )
        if file_path:
            self.model_path = file_path
            self.model_path_field.setText(file_path)
            try:
                with open(self.LAST_MODEL_FILE, "w") as f:
                    f.write(file_path)
            except Exception:
                pass
            
    def browse_input_transcript(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Transcript", "", "Text Files (*.txt)"
        )
        if file_path:
            self.input_transcript = file_path
            self.input_transcript_field.setText(file_path)

            base_name, ext = os.path.splitext(file_path)
            timestamp = datetime.datetime.now().strftime("%I%M%p").lower()
            output_path = f"{base_name}_processed_{timestamp}{ext}"
            self.output_transcript = output_path
            self.output_transcript_field.setText(output_path)
            self.processed_transcript_path = output_path
            self.audio_transcript_field.setText(output_path)
            self.save_paths()
            
    def browse_output_transcript(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Transcript", "", "Text Files (*.txt)"
        )
        if file_path:
            base, ext = os.path.splitext(file_path)
            timestamp = datetime.datetime.now().strftime("%I%M%p").lower()
            file_path = f"{base}_processed_{timestamp}{ext}"
            self.output_transcript = file_path
            self.output_transcript_field.setText(file_path)
            self.processed_transcript_path = file_path
            self.audio_transcript_field.setText(file_path)
            self.save_paths()

    def browse_audio_transcript(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Processed Transcript", "", "Text Files (*.txt)"
        )
        if file_path:
            self.processed_transcript_path = file_path
            self.audio_transcript_field.setText(file_path)
            self.save_paths()

    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.mov *.mkv)"
        )
        if file_path:
            self.input_video_path = file_path
            self.video_path_field.setText(file_path)
            base, _ = os.path.splitext(file_path)
            self.output_video_path = f"{base}_videocut.mp4"
            self.output_video_field.setText(self.output_video_path)
            self.save_paths()

    def browse_output_video(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Video", "", "Video Files (*.mp4 *.mov *.mkv)"
        )
        if file_path:
            self.output_video_path = file_path
            self.output_video_field.setText(self.output_video_path)
            self.save_paths()
            
    def process_transcript(self):
        # Validate inputs
        if not self.model_path_field.text():
            QMessageBox.warning(self, "No Model", "Please select a valid model file.")
            return
            
        if not self.input_transcript_field.text():
            QMessageBox.warning(self, "No Input", "Please select a valid input transcript file.")
            return
            
        if not self.output_transcript_field.text():
            QMessageBox.warning(self, "No Output", "Please specify an output transcript file.")
            return
        
        # Get parameters
        model_path = self.model_path_field.text()
        input_path = self.input_transcript_field.text()
        base, ext = os.path.splitext(input_path)
        timestamp = datetime.datetime.now().strftime("%I%M%p").lower()
        output_path = f"{base}_processed_{timestamp}{ext}"
        self.output_transcript_field.setText(output_path)
        self.processed_transcript_path = output_path
        self.audio_transcript_field.setText(output_path)
        self.save_paths()
        threshold = self.threshold_spinner.value()
        keep_ratio = self.keep_ratio_spinner.value()
        
        # Clear log
        self.process_log_viewer.clear()
        self.add_process_log("Starting transcript processing...")
        
        # Disable button
        self.process_btn.setEnabled(False)
        
        # Run processing in background thread
        self.process_thread = WorkerThread(
            process_transcript,
            model_path,
            input_path,
            output_path=output_path,
            threshold=threshold,
            keep_ratio=keep_ratio,
        )
        self.process_thread.progress_update.connect(self.add_process_log)
        self.process_thread.work_finished.connect(self.on_processing_finished)
        self.process_thread.start()

    def open_last_output_transcript(self):
        """Open the most recently processed transcript using gnome-text-editor"""
        path = self.processed_transcript_path or self.output_transcript_field.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "No output transcript available to open.")
            return
        try:
            subprocess.Popen(["gnome-text-editor", path])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open transcript: {str(e)}")

    def process_video(self):
        if not self.processed_transcript_path:
            QMessageBox.warning(self, "No Transcript", "Please select a processed transcript file.")
            return
        if not self.input_video_path:
            QMessageBox.warning(self, "No Video", "Please select an input video file.")
            return
        if not self.output_video_field.text():
            QMessageBox.warning(self, "No Output", "Please specify an output video file.")
            return

        transcript_path = self.processed_transcript_path
        video_path = self.input_video_path
        output_path = self.output_video_path
        self.output_video_field.setText(output_path)
        self.save_paths()

        self.audio_log_viewer.clear()
        self.add_audio_log("Starting video cutting...")
        self.audio_process_btn.setEnabled(False)

        self.video_thread = WorkerThread(
            cut_video,
            video_path,
            transcript_path,
            output_path,
        )
        self.video_thread.progress_update.connect(self.add_audio_log)
        self.video_thread.work_finished.connect(self.on_video_finished)
        self.video_thread.start()
    
    def add_log_message(self, message):
        """Add a message to the training log viewer with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_viewer.append(f"[{timestamp}] {message}")
        
        # Update progress bar based on message content
        if "Epoch" in message and "/" in message:
            try:
                # Try to extract progress information
                parts = message.split()
                for i, part in enumerate(parts):
                    if part.startswith("Epoch"):
                        epoch_info = parts[i+1] if i+1 < len(parts) else ""
                        if "/" in epoch_info:
                            current, total = epoch_info.split("/")
                            try:
                                current = int(current)
                                total = int(total.rstrip(","))
                                progress = int((current / total) * 100)
                                self.progress_bar.setValue(progress)
                            except (ValueError, ZeroDivisionError):
                                pass
                        break
            except Exception:
                # Ignore any parsing errors
                pass
        
        # Scroll to bottom
        scrollbar = self.log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Process events to update UI
        QApplication.processEvents()
    
    def add_process_log(self, message):
        """Add a message to the processing log viewer with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.process_log_viewer.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.process_log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Process events to update UI
        QApplication.processEvents()

    def add_audio_log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.audio_log_viewer.append(f"[{timestamp}] {message}")
        scrollbar = self.audio_log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    def on_video_finished(self, success, message):
        self.audio_process_btn.setEnabled(True)
        if success:
            self.add_audio_log("Video cutting completed!")
            QMessageBox.information(self, "Success", "Video saved successfully!")
            self.save_paths()
        else:
            self.add_audio_log(f"Video cutting failed: {message}")
            QMessageBox.critical(self, "Error", f"Video cutting failed: {message}")
        
    def on_training_finished(self, success, message):
        # Re-enable button
        self.train_btn.setEnabled(True)
        
        if success:
            self.progress_bar.setValue(100)
            self.add_log_message("Training completed successfully!")
            QMessageBox.information(self, "Success", "Model training completed successfully!")
            self.model_path = self.model_path_edit.text()
            self.model_path_field.setText(self.model_path)
            try:
                with open(self.LAST_MODEL_FILE, "w") as f:
                    f.write(self.model_path)
            except Exception:
                pass
            self.save_paths()
        else:
            self.add_log_message(f"Training failed: {message}")
            QMessageBox.critical(self, "Error", f"Training failed: {message}")
            
    def on_processing_finished(self, success, message):
        # Re-enable button
        self.process_btn.setEnabled(True)
        
        if success:
            self.add_process_log("Processing completed successfully!")
            QMessageBox.information(self, "Success", "Transcript processing completed successfully!")
            self.processed_transcript_path = self.output_transcript_field.text()
            self.audio_transcript_field.setText(self.processed_transcript_path)
            self.save_paths()
        else:
            self.add_process_log(f"Processing failed: {message}")
            QMessageBox.critical(self, "Error", f"Processing failed: {message}")

    def load_full_scripts_data(self):
        """Hardcoded function to load data from GOODFULLSCRIPTSDATA folder"""
        data_path = "/home/j/Desktop/code/pytorchSilencer/GOODFULLSCRIPTSDATA"
        if os.path.exists(data_path) and os.path.isdir(data_path):
            self.good_data_dir = data_path
            self.good_data_path.setText(data_path)
            self.update_data_status()
            count = sum(1 for f in os.listdir(data_path) if f.endswith('.txt'))
            self.add_log_message(f"Loaded Full Scripts data with {count} transcript files")
        else:
            QMessageBox.warning(self, "Error", "GOODFULLSCRIPTSDATA folder not found at expected location")

    def load_paths(self):
        """Load previously used paths from disk if available"""
        if os.path.exists(self.PATHS_FILE):
            try:
                with open(self.PATHS_FILE, "r") as f:
                    data = json.load(f)
                self.good_data_dir = data.get("good_data_dir", self.good_data_dir)
                self.bad_data_dir = data.get("bad_data_dir", self.bad_data_dir)
                self.input_transcript = data.get("input_transcript", self.input_transcript)
                self.output_transcript = data.get("output_transcript", self.output_transcript)
                self.processed_transcript_path = data.get("processed_transcript_path", self.processed_transcript_path)
                self.input_video_path = data.get("input_video_path", self.input_video_path)
                self.output_video_path = data.get("output_video_path", self.output_video_path)
            except Exception:
                pass

    def save_paths(self):
        """Persist current paths to disk"""
        data = {
            "good_data_dir": self.good_data_dir,
            "bad_data_dir": self.bad_data_dir,
            "input_transcript": self.input_transcript,
            "output_transcript": self.output_transcript,
            "processed_transcript_path": self.processed_transcript_path,
            "input_video_path": self.input_video_path,
            "output_video_path": self.output_video_path,
        }
        try:
            with open(self.PATHS_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def closeEvent(self, event):
        """Save paths when the application closes"""
        self.save_paths()
        super().closeEvent(event)

def main():
    # Create application directory structure
    os.makedirs("models", exist_ok=True)
    os.makedirs("sample_data", exist_ok=True)
    
    try:
        app = QApplication(sys.argv)
        window = PytorchSilencerApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"Application error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
