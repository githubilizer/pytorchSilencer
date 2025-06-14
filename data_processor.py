import re
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

class TranscriptEntry:
    def __init__(self, start_time: float, end_time: float, text: str, silence_after: float = 0.0):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.silence_after = silence_after  # Duration of silence after this word
    
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
        """Convert transcript to feature matrix for model input
        Enhanced to provide better context for LSTM models
        """
        features = []
        
        for i, entry in enumerate(self.entries[:-1]):  # Skip last entry as it has no next word
            # Skip entries that are just silence
            if entry.text.strip() == '':
                continue
                
            # Calculate silence duration after word
            silence_duration = self.entries[i+1].start_time - entry.end_time
            
            # Get preceding word properties (context)
            prev_word_len = 0
            prev_word_duration = 0
            prev_silence_duration = 0
            
            if i > 0 and self.entries[i-1].text.strip():
                prev_word_len = len(self.entries[i-1].text)
                prev_word_duration = self.entries[i-1].end_time - self.entries[i-1].start_time
                prev_silence_duration = entry.start_time - self.entries[i-1].end_time
            
            # Get next word properties (if available)
            next_word_len = 0
            next_word_duration = 0
            
            if i+1 < len(self.entries) and self.entries[i+1].text.strip():
                next_word_len = len(self.entries[i+1].text)
                next_word_duration = self.entries[i+1].end_time - self.entries[i+1].start_time
            
            # Create features for this entry
            entry_features = [
                len(entry.text),                       # Word length
                entry.end_time - entry.start_time,     # Word duration
                silence_duration,                      # Silence duration after word
                prev_word_len,                         # Previous word length
                prev_word_duration,                    # Previous word duration
                prev_silence_duration,                 # Silence before current word
                next_word_len,                         # Next word length (if available)
                next_word_duration,                    # Next word duration
                1 if ',' in entry.text else 0,         # Comma presence
                1 if '.' in entry.text else 0,         # Period presence
                1 if '?' in entry.text else 0,         # Question mark presence
                1 if '!' in entry.text else 0,         # Exclamation mark presence
                entry.start_time,                      # Absolute start time in audio
                float(i) / len(self.entries)           # Relative position in transcript
            ]
            
            features.append(entry_features)
            
        return np.array(features) if features else np.array([])
    
    def get_contextual_sequences(self, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate contextual sequences for LSTM processing
        
        Args:
            sequence_length: Length of each sequence
            stride: Step size for sliding window
            
        Returns:
            Tuple of (features, silences) where each is a 3D array of shape 
            [num_sequences, sequence_length, num_features]
        """
        features = self.to_features()
        if len(features) == 0:
            return np.array([]), np.array([])
            
        # Calculate silence durations (labels)
        silences = []
        for i in range(len(self.entries) - 1):
            if self.entries[i].text.strip() == '':
                continue
                
            next_entry = self.entries[i+1]
            silence_duration = next_entry.start_time - self.entries[i].end_time
            silences.append(silence_duration)
        
        # Generate sequences
        feature_sequences = []
        silence_sequences = []
        
        for i in range(0, len(features) - sequence_length + 1, stride):
            feature_seq = features[i:i+sequence_length]
            silence_seq = silences[i:i+sequence_length]
            
            # Pad sequences if needed
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
        """Parse a transcript file with timestamps into a TranscriptData object"""
        transcript = TranscriptData()
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Parse timestamp line
            timestamp_match = re.match(r'\[([\d\.]+) -> ([\d\.]+)\] (.*)', line)
            if timestamp_match:
                start_time = float(timestamp_match.group(1))
                end_time = float(timestamp_match.group(2))
                text = timestamp_match.group(3)
                
                # Add the entry
                entry = TranscriptEntry(start_time, end_time, text)
                transcript.add_entry(entry)
            
            i += 1
            
        # Calculate silence durations
        for i in range(len(transcript.entries) - 1):
            current = transcript.entries[i]
            next_entry = transcript.entries[i+1]
            
            # If the next entry is not empty text (which represents silence)
            if next_entry.text.strip():
                current.silence_after = next_entry.start_time - current.end_time
        
        return transcript

    @staticmethod
    def estimate_typical_silence(transcript: TranscriptData, max_duration: float = 1.0) -> float:
        """Estimate typical short silence duration within a transcript."""
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
        """Save transcript with silence cut markers.

        If ``cut_durations`` is provided, each ``[SILENCE-CUT]`` line will
        include the amount of silence to remove from that gap.
        """
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
                                    f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] "
                                    f"[SILENCE-CUT {cut:.2f}s]\n"
                                )
                            else:
                                f.write(
                                    f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] [SILENCE-CUT]\n"
                                )
                        else:
                            f.write(f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] \n")
                            
        print(f"Processed transcript saved to {output_path}")
    
    @staticmethod
    def load_training_data(directory: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all training transcripts from a directory"""
        all_features = []
        all_labels = []  # 1 for silence to keep, 0 for silence to cut
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                transcript = TranscriptProcessor.parse_transcript(file_path)
                
                # Extract features
                features = transcript.to_features()
                
                if len(features) > 0:
                    # For training data, we assume all silences are good (should be kept)
                    labels = np.ones(len(features))
                    
                    all_features.append(features)
                    all_labels.append(labels)
        
        # Concatenate all data
        if all_features:
            return np.vstack(all_features), np.concatenate(all_labels)
        else:
            return np.array([]), np.array([])
    
    @staticmethod
    def load_training_sequences(directory: str, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Load all training transcripts as sequences for LSTM training"""
        all_feature_sequences = []
        all_silence_sequences = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                transcript = TranscriptProcessor.parse_transcript(file_path)
                
                # Extract sequential features
                feature_seqs, silence_seqs = transcript.get_contextual_sequences(sequence_length, stride)
                
                if len(feature_seqs) > 0:
                    all_feature_sequences.append(feature_seqs)
                    # For training data, we assume all silences are good (should be kept)
                    all_silence_sequences.append(np.ones_like(silence_seqs))
        
        # Concatenate all data
        if all_feature_sequences:
            return np.vstack(all_feature_sequences), np.vstack(all_silence_sequences)
        else:
            return np.array([]), np.array([])

    @staticmethod
    def load_duration_sequences(directory: str, sequence_length: int = 10, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Load training transcripts returning actual silence durations.

        This is used for models that predict the desired silence length for each
        pause rather than a simple keep/cut decision.
        """
        all_feature_sequences = []
        all_duration_sequences = []

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                transcript = TranscriptProcessor.parse_transcript(file_path)

                feature_seqs, duration_seqs = transcript.get_contextual_sequences(sequence_length, stride)

                if len(feature_seqs) > 0:
                    all_feature_sequences.append(feature_seqs)
                    all_duration_sequences.append(duration_seqs)

        if all_feature_sequences:
            return np.vstack(all_feature_sequences), np.vstack(all_duration_sequences)
        else:
            return np.array([]), np.array([])
