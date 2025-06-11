import os
import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class SilenceAutoencoder(nn.Module):
    """Simple feedforward autoencoder for silence features."""

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderPredictor:
    """Train an autoencoder on good silences and flag anomalies."""

    def __init__(self, model_path: Optional[str] = None, input_size: int = 14, hidden_size: int = 32,
                 force_cpu: bool = False):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
        self.model = SilenceAutoencoder(input_size, hidden_size).to(self.device)
        self.scaler = StandardScaler()
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, features: np.ndarray, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> List[float]:
        """Train the autoencoder."""
        self.scaler.fit(features)
        X = torch.tensor(self.scaler.transform(features), dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))
        return losses

    def reconstruction_errors(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = torch.tensor(self.scaler.transform(features), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(X)
            err = torch.mean((recon - X) ** 2, dim=1)
        return err.cpu().numpy()

    def predict(self, features: np.ndarray, threshold: float = 0.01) -> List[bool]:
        """Return True to keep silence, False to cut."""
        if len(features) == 0:
            return []
        errors = self.reconstruction_errors(features)
        return [e <= threshold for e in errors]

    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'input_size': self.input_size}, model_path)
        with open(model_path + '.scaler', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_size = checkpoint.get('input_size', self.input_size)
        self.model = SilenceAutoencoder(self.input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        scaler_path = model_path + '.scaler'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

