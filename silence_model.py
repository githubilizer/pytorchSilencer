import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict
import os
import pickle

def get_device_info():
    """Get detailed info about available CUDA devices"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if info["cuda_available"]:
        for i in range(info["device_count"]):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory": torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            }
            info["devices"].append(device_info)
    
    return info

class LSTMSilenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMSilenceClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_size)
        
        if lengths is not None:
            # Pack padded sequence for more efficient computation
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Run LSTM
            lstm_out, _ = self.lstm(x_packed)
            
            # Unpack the sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            # Run LSTM without packing (all sequences same length)
            lstm_out, _ = self.lstm(x)
        
        # Apply fully connected layers to each time step
        batch_size, seq_len, _ = lstm_out.size()
        
        # Reshape for FC layers: (batch_size * seq_len, hidden_size * 2)
        fc_in = lstm_out.contiguous().view(batch_size * seq_len, self.hidden_size * 2)
        
        # Apply FC layers
        fc_out = self.fc(fc_in)
        
        # Reshape back to (batch_size, seq_len, 1)
        outputs = fc_out.view(batch_size, seq_len, 1)

        return outputs

class LSTMSilenceRegressor(nn.Module):
    """Predict the ideal remaining silence duration for each pause."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        batch_size, seq_len, _ = lstm_out.size()
        fc_in = lstm_out.contiguous().view(batch_size * seq_len, self.hidden_size * 2)
        fc_out = self.fc(fc_in)
        outputs = fc_out.view(batch_size, seq_len, 1)
        return outputs

class SequenceScaler:
    """Custom scaler for 3D sequence data"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X):
        """
        Fit scaler to 3D sequence data or 2D feature data
        X shape: (batch_size, seq_len, features) or (n_samples, n_features)
        """
        # Check if input is 2D or 3D
        if len(X.shape) == 2:
            # Already 2D, no need to reshape
            X_reshaped = X
        else:
            # Reshape 3D to 2D for StandardScaler
            batch_size, seq_len, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
        
        # Fit scaler
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform 3D sequence data or 2D feature data
        X shape: (batch_size, seq_len, features) or (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet.")
            
        # Get original shape
        original_shape = X.shape
        is_2d = len(original_shape) == 2
        
        # Reshape if needed
        if is_2d:
            X_reshaped = X
        else:
            # Reshape 3D to 2D for transformation
            X_reshaped = X.reshape(-1, original_shape[2])
        
        # Transform
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to original shape if needed
        if not is_2d:
            X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled

class SilencePredictor:
    def __init__(self, model_path: Optional[str] = None, sequence_length: int = 10,
                 input_size: int = 14, force_cpu: bool = False,
                 long_silence_threshold: Optional[float] = None):
        self.input_size = input_size  # Updated feature count
        self.sequence_length = sequence_length
        
        # CUDA detection and device selection
        self.cuda_info = get_device_info()
        
        # Log CUDA information
        if self.cuda_info["cuda_available"]:
            for device in self.cuda_info["devices"]:
                print(f"Found CUDA device: {device['name']} (Compute capability: {device['capability'][0]}.{device['capability'][1]}, "
                      f"Memory: {device['memory']:.2f} GB)")
                      
        # Determine device - use CUDA if available and not forced to CPU
        if not force_cpu and self.cuda_info["cuda_available"]:
            # RTX 5060 should be supported with compute capability 8.9
            # Try to find an appropriate device (preferring higher compute capability)
            best_device_idx = 0
            best_capability = (0, 0)
            
            for i, device in enumerate(self.cuda_info["devices"]):
                if device["capability"] > best_capability:
                    best_capability = device["capability"]
                    best_device_idx = i
            
            self.device = torch.device(f"cuda:{best_device_idx}")
            print(f"Using CUDA device: {torch.cuda.get_device_name(best_device_idx)}")
        else:
            self.device = torch.device("cpu")
            if force_cpu:
                print("Forcing CPU usage as requested")
            else:
                print("CUDA not available, using CPU")
        
        # Initialize model
        self.model = LSTMSilenceClassifier(self.input_size)
        self.scaler = SequenceScaler()
        self.long_silence_threshold = long_silence_threshold
        
        # Move model to device
        self.model.to(self.device)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              epochs: int = 100, batch_size: int = 8, 
              learning_rate: float = 0.001) -> List[float]:
        """Train the LSTM silence classifier model"""
        # Log device being used
        print(f"Training on device: {self.device}")
        
        # Fit scaler on sequence features
        print(f"Scaling features...")
        self.scaler.fit(features)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Convert to PyTorch tensors
        print(f"Converting to tensors and moving to {self.device}...")
        X = torch.FloatTensor(scaled_features).to(self.device)
        y = torch.FloatTensor(labels).to(self.device).unsqueeze(-1)
        
        # Define loss function and optimizer
        print(f"Setting up training...")
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            # Print the start of the epoch
            print(f"Epoch {epoch+1}/{epochs} - Training in progress...")
            
            # Shuffle indices
            indices = torch.randperm(X.size(0))
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for start_idx in range(0, X.size(0), batch_size):
                end_idx = min(start_idx + batch_size, X.size(0))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward pass - no need for lengths as all sequences have same length
                outputs = self.model(X_batch)
                
                # Calculate loss
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print batch progress for large datasets (every 10 batches)
                if X.size(0) > 100 and (num_batches % 10 == 0 or num_batches == 1):
                    print(f"  Batch {num_batches}/{(X.size(0) + batch_size - 1) // batch_size} processed")
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} complete, Loss: {avg_loss:.6f}")
        
        print(f"Training completed with {epochs} epochs")
        return losses
    
    def predict_scores(self, features: np.ndarray) -> List[float]:
        """Return raw prediction scores for each silence."""
        # Log device being used
        print(f"Predicting on device: {self.device}")

        if len(features) == 0:
            return []
            
        # Prepare sequences for LSTM input
        # We need to reshape the features into sequences
        seq_length = self.sequence_length
        n_features = features.shape[1]
        
        # Pad if necessary
        if len(features) < seq_length:
            padding = np.zeros((seq_length - len(features), n_features))
            features_padded = np.vstack([features, padding])
            sequences = np.array([features_padded])
        else:
            # Create overlapping sequences
            n_sequences = len(features) - seq_length + 1
            sequences = np.array([features[i:i+seq_length] for i in range(n_sequences)])
        
        # Scale features
        try:
            scaled_sequences = self.scaler.transform(sequences)
        except ValueError:
            # If scaler not fitted, just normalize
            means = np.mean(sequences, axis=(0, 1), keepdims=True)
            stds = np.std(sequences, axis=(0, 1), keepdims=True) + 1e-8
            scaled_sequences = (sequences - means) / stds
        
        # Convert to PyTorch tensor
        X = torch.FloatTensor(scaled_sequences).to(self.device)
        
        # Predict
        self.model.eval()
        scores = []
        
        try:
            with torch.no_grad():
                outputs = self.model(X)  # Shape: [batch_size, seq_len, 1]
                outputs = outputs.squeeze(-1).cpu().numpy()
                
                # Process the overlapping predictions
                if len(features) <= seq_length:
                    # For short sequences, just return all predictions
                    scores = outputs[0, :len(features)]
                else:
                    # For longer sequences, we have overlapping predictions
                    # Initialize array to accumulate predictions
                    accumulated = np.zeros(len(features))
                    counts = np.zeros(len(features))
                    
                    # Aggregate predictions from overlapping sequences
                    for i in range(len(outputs)):
                        for j in range(seq_length):
                            idx = i + j
                            if idx < len(features):
                                accumulated[idx] += outputs[i, j]
                                counts[idx] += 1
                    
                    # Average predictions
                    for i in range(len(features)):
                        if counts[i] > 0:
                            scores.append(accumulated[i] / counts[i])
                        else:
                            scores.append(1.0)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error. Falling back to CPU...")
                # Move model to CPU and try again
                backup_device = self.device
                self.device = torch.device("cpu")
                self.model.to(self.device)
                X = X.to(self.device)
                
                # Retry prediction
                with torch.no_grad():
                    outputs = self.model(X)
                    outputs = outputs.squeeze(-1).cpu().numpy()
                    
                    # Same processing as above
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
                
                # Move model back to original device
                self.device = backup_device
                self.model.to(self.device)
            else:
                # For other errors, just re-raise
                raise
        
        return [float(p) for p in scores]

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> List[bool]:
        """Predict which silences should be cut (False) or kept (True)."""
        scores = self.predict_scores(features)
        return [s >= threshold for s in scores]

    def predict_with_scores(
        self, features: np.ndarray, threshold: float = 0.5
    ) -> Tuple[List[bool], List[float]]:
        """Return boolean predictions along with raw scores."""
        scores = self.predict_scores(features)
        preds = [s >= threshold for s in scores]
        return preds, scores
    
    def save_model(self, model_path: str):
        """Save model and scaler to file"""
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'input_size': self.input_size,
            'long_silence_threshold': self.long_silence_threshold,
        }, model_path)
        
        # Save scaler
        scaler_path = model_path + ".scaler"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load model and scaler from file"""
        print(f"Loading model from {model_path} to device: {self.device}")
        
        # Load model state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Update input size if it's in the checkpoint
        if 'input_size' in checkpoint:
            self.input_size = checkpoint['input_size']
            # Recreate model with correct input size
            self.model = LSTMSilenceClassifier(self.input_size)
            self.model.to(self.device)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.sequence_length = checkpoint.get('sequence_length', 10)  # Default to 10 if not found
        self.long_silence_threshold = checkpoint.get('long_silence_threshold')
        self.model.eval()
        
        # Load scaler
        scaler_path = model_path + ".scaler"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
        print(f"Model loaded from {model_path}")


class SilenceDurationPredictor:
    """Predict how much silence should remain after each word."""

    def __init__(self, model_path: Optional[str] = None, sequence_length: int = 10,
                 input_size: int = 14, force_cpu: bool = False,
                 long_silence_threshold: Optional[float] = None):
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

    def train(self, features: np.ndarray, durations: np.ndarray,
              epochs: int = 100, batch_size: int = 8,
              learning_rate: float = 0.001) -> List[float]:
        self.scaler.fit(features)
        X = torch.FloatTensor(self.scaler.transform(features)).to(self.device)
        y = torch.FloatTensor(durations).to(self.device).unsqueeze(-1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
            sequences = np.array([features[i:i+seq_length] for i in range(n_sequences)])

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
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'input_size': self.input_size,
            'long_silence_threshold': self.long_silence_threshold,
            'duration_model': True,
        }, model_path)
        with open(model_path + '.scaler', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.sequence_length = checkpoint.get('sequence_length', 10)
        self.input_size = checkpoint.get('input_size', self.input_size)
        self.long_silence_threshold = checkpoint.get('long_silence_threshold')
        self.model = LSTMSilenceRegressor(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        scaler_path = model_path + '.scaler'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)


# Run a quick test to verify CUDA is working
if __name__ == "__main__":
    print("Testing CUDA availability:")
    cuda_info = get_device_info()
    
    if cuda_info["cuda_available"]:
        print(f"CUDA is available with {cuda_info['device_count']} device(s):")
        for i, device in enumerate(cuda_info["devices"]):
            print(f"  Device {i}: {device['name']} (Capability: {device['capability'][0]}.{device['capability'][1]})")
        
        # Create a small test tensor
        print("\nRunning a quick CUDA test...")
        device = torch.device("cuda:0")
        x = torch.rand(100, 100).to(device)
        y = torch.matmul(x, x)
        print(f"Test completed successfully on {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available on this system.") 
