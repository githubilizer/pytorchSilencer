import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QTextEdit, QTabWidget, 
                            QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, 
                            QCheckBox, QGroupBox, QSplitter, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont

from data_processor import TranscriptProcessor
from silence_model import SilencePredictor

class SilenceTrainerThread(QThread):
    progress_update = pyqtSignal(int, float, str)  # Added string parameter for log message
    training_finished = pyqtSignal(list)
    
    def __init__(self, predictor, features, labels, epochs, batch_size, sequence_mode=True):
        super().__init__()
        self.predictor = predictor
        self.features = features
        self.labels = labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []
        self.sequence_mode = sequence_mode
        self.start_time = time.time()
        
    def get_elapsed_time(self):
        """Get formatted elapsed time"""
        elapsed = time.time() - self.start_time
        return str(datetime.timedelta(seconds=int(elapsed)))
        
    def run(self):
        losses = []
        total_batches = self.epochs * 10  # For progress calculation
        
        # Log initial message
        self.progress_update.emit(0, 0.0, f"Starting training with {len(self.features)} sequences...")
        
        # For sequence mode, we just call train directly
        if self.sequence_mode:
            # Custom training loop for better progress reporting
            try:
                # Fit scaler on sequence features
                self.progress_update.emit(1, 0.0, "Scaling features...")
                self.predictor.scaler.fit(self.features)
                
                # Scale features
                scaled_features = self.predictor.scaler.transform(self.features)
                
                # Convert to PyTorch tensors
                self.progress_update.emit(2, 0.0, "Converting to tensors...")
                import torch
                X = torch.FloatTensor(scaled_features).to(self.predictor.device)
                y = torch.FloatTensor(self.labels).to(self.predictor.device).unsqueeze(-1)
                
                # Define loss function and optimizer
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.Adam(self.predictor.model.parameters(), lr=0.001)
                
                # Training loop
                self.predictor.model.train()
                
                for epoch in range(self.epochs):
                    epoch_start = time.time()
                    self.progress_update.emit(
                        int((epoch / self.epochs) * 100), 
                        0.0, 
                        f"Epoch {epoch+1}/{self.epochs} started... [Time: {self.get_elapsed_time()}]"
                    )
                    
                    # Shuffle indices
                    indices = torch.randperm(X.size(0))
                    
                    epoch_loss = 0.0
                    num_batches = 0
                    
                    # Process in batches
                    for start_idx in range(0, X.size(0), self.batch_size):
                        batch_num = start_idx // self.batch_size + 1
                        total_batches = (X.size(0) + self.batch_size - 1) // self.batch_size
                        
                        if batch_num % 10 == 0 or batch_num == 1 or batch_num == total_batches:
                            self.progress_update.emit(
                                int((epoch / self.epochs) * 100) + int((batch_num / total_batches) * (100 / self.epochs)),
                                epoch_loss / max(1, num_batches),
                                f"Epoch {epoch+1}/{self.epochs}, Batch {batch_num}/{total_batches} [Time: {self.get_elapsed_time()}]"
                            )
                        
                        end_idx = min(start_idx + self.batch_size, X.size(0))
                        batch_indices = indices[start_idx:end_idx]
                        
                        # Get batch data
                        X_batch = X[batch_indices]
                        y_batch = y[batch_indices]
                        
                        # Forward pass - no need for lengths as all sequences have same length
                        outputs = self.predictor.model(X_batch)
                        
                        # Calculate loss
                        loss = criterion(outputs, y_batch)
                        
                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    
                    avg_loss = epoch_loss / num_batches
                    losses.append(avg_loss)
                    self.losses = losses
                    
                    epoch_time = time.time() - epoch_start
                    self.progress_update.emit(
                        int(((epoch + 1) / self.epochs) * 100),
                        avg_loss,
                        f"Epoch {epoch+1}/{self.epochs} completed in {epoch_time:.1f}s - Loss: {avg_loss:.6f} [Time: {self.get_elapsed_time()}]"
                    )
                
                self.training_finished.emit(losses)
                return
            
            except Exception as e:
                self.progress_update.emit(0, 0.0, f"Error during training: {str(e)}")
                import traceback
                traceback.print_exc()
                return
                
            # Original version (keeping as fallback)
            batch_losses = self.predictor.train(
                self.features, self.labels, 
                epochs=self.epochs, 
                batch_size=self.batch_size
            )
            
            # Update progress in chunks
            chunk_size = max(1, len(batch_losses) // 100)
            for i in range(0, len(batch_losses), chunk_size):
                chunk = batch_losses[i:i+chunk_size]
                avg_loss = sum(chunk) / len(chunk)
                progress = min(100, int((i / len(batch_losses)) * 100))
                self.losses = batch_losses[:i+chunk_size]
                
                # Add timestamp to progress message
                timestamp = self.get_elapsed_time()
                self.progress_update.emit(
                    progress, 
                    avg_loss, 
                    f"Progress: {progress}% - Loss: {avg_loss:.6f} [Time: {timestamp}]"
                )
                
            self.training_finished.emit(batch_losses)
            return
            
        # Legacy non-sequence mode
        for epoch in range(self.epochs):
            # Here we would normally train for one epoch, but our trainer doesn't support
            # progress callbacks, so we'll simulate it by slicing the data
            slice_size = len(self.features) // 10  # 10 slices per epoch
            for i in range(10):
                start_idx = i * slice_size
                end_idx = min((i + 1) * slice_size, len(self.features))
                
                X_slice = self.features[start_idx:end_idx]
                y_slice = self.labels[start_idx:end_idx]
                
                # Train on this slice (in reality, we would have a proper batch training method)
                epoch_loss = self.predictor.train(X_slice, y_slice, epochs=1, batch_size=self.batch_size)[0]
                losses.append(epoch_loss)
                self.losses = losses
                
                # Update progress with timestamp
                progress = (epoch * 10 + i + 1) / (self.epochs * 10) * 100
                timestamp = self.get_elapsed_time()
                self.progress_update.emit(
                    int(progress), 
                    epoch_loss, 
                    f"Epoch {epoch+1}/{self.epochs}, Slice {i+1}/10 - Loss: {epoch_loss:.6f} [Time: {timestamp}]"
                )
        
        self.training_finished.emit(losses)

class PytorchSilencerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch Silencer (LSTM)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark mode
        self.setup_dark_theme()
        
        # Initialize model
        self.predictor = SilencePredictor()
        self.training_data_dir = ""
        self.model_path = "models/silence_model.pt"
        self.current_transcript = None
        self.cut_predictions = []
        
        # Setup UI
        self.setup_ui()
    
    def setup_dark_theme(self):
        # Set dark palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(dark_palette)
        QApplication.setStyle("Fusion")
        
        # Set stylesheet for additional customization
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #353535;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #2D5C7F;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3A7CA5;
            }
            QPushButton:pressed {
                background-color: #1F3F5F;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
            QTextEdit, QLineEdit {
                background-color: #252525;
                color: #EEEEEE;
                border: 1px solid #555555;
                border-radius: 2px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #303030;
            }
            QTabBar::tab {
                background-color: #303030;
                color: #CCCCCC;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid #555555;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #404040;
                color: white;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2D5C7F;
                width: 10px;
            }
        """)
    
    def setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Training tab
        training_tab = QWidget()
        tabs.addTab(training_tab, "Training")
        self.setup_training_tab(training_tab)
        
        # Prediction tab
        prediction_tab = QWidget()
        tabs.addTab(prediction_tab, "Prediction")
        self.setup_prediction_tab(prediction_tab)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def setup_training_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Data selection section
        data_group = QGroupBox("Training Data")
        data_layout = QVBoxLayout(data_group)
        
        data_path_layout = QHBoxLayout()
        self.training_data_path = QLineEdit()
        self.training_data_path.setPlaceholderText("Path to training data directory...")
        self.training_data_path.setReadOnly(True)
        data_path_layout.addWidget(self.training_data_path)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_training_data)
        data_path_layout.addWidget(browse_btn)
        data_layout.addLayout(data_path_layout)
        
        self.data_status_label = QLabel("No data loaded")
        data_layout.addWidget(self.data_status_label)
        
        layout.addWidget(data_group)
        
        # LSTM parameters section
        lstm_group = QGroupBox("LSTM Parameters")
        lstm_layout = QHBoxLayout(lstm_group)
        
        lstm_layout.addWidget(QLabel("Sequence Length:"))
        self.seq_length_spinner = QSpinBox()
        self.seq_length_spinner.setRange(1, 50)
        self.seq_length_spinner.setValue(10)
        lstm_layout.addWidget(self.seq_length_spinner)
        
        lstm_layout.addWidget(QLabel("Stride:"))
        self.stride_spinner = QSpinBox()
        self.stride_spinner.setRange(1, 10)
        self.stride_spinner.setValue(1)
        lstm_layout.addWidget(self.stride_spinner)
        
        layout.addWidget(lstm_group)
        
        # Training parameters section
        params_group = QGroupBox("Training Parameters")
        params_layout = QHBoxLayout(params_group)
        
        params_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spinner = QSpinBox()
        self.epochs_spinner.setRange(1, 1000)
        self.epochs_spinner.setValue(50)
        params_layout.addWidget(self.epochs_spinner)
        
        params_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 32)
        self.batch_size_spinner.setValue(8)
        params_layout.addWidget(self.batch_size_spinner)
        
        params_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spinner = QDoubleSpinBox()
        self.lr_spinner.setRange(0.0001, 0.1)
        self.lr_spinner.setValue(0.001)
        self.lr_spinner.setDecimals(4)
        self.lr_spinner.setSingleStep(0.0001)
        params_layout.addWidget(self.lr_spinner)
        
        layout.addWidget(params_group)
        
        # Training progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.loss_label = QLabel("Loss: N/A")
        progress_layout.addWidget(self.loss_label)
        
        # Add a log viewer
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Training Log:"))
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(100)
        log_layout.addWidget(self.log_viewer)
        progress_layout.addLayout(log_layout)
        
        # Matplotlib setup for loss plotting
        self.figure = plt.figure(figsize=(5, 3))
        plt.style.use('dark_background')
        self.canvas = FigureCanvas(self.figure)
        progress_layout.addWidget(self.canvas)
        
        layout.addWidget(progress_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        buttons_layout.addWidget(self.train_btn)
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        buttons_layout.addWidget(self.load_model_btn)
        
        layout.addLayout(buttons_layout)
    
    def setup_prediction_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Transcript selection
        transcript_group = QGroupBox("Transcript")
        transcript_layout = QVBoxLayout(transcript_group)
        
        file_layout = QHBoxLayout()
        self.transcript_path = QLineEdit()
        self.transcript_path.setPlaceholderText("Path to transcript file...")
        self.transcript_path.setReadOnly(True)
        file_layout.addWidget(self.transcript_path)
        
        browse_transcript_btn = QPushButton("Browse")
        browse_transcript_btn.clicked.connect(self.browse_transcript)
        file_layout.addWidget(browse_transcript_btn)
        transcript_layout.addLayout(file_layout)
        
        # Transcript viewer
        self.transcript_viewer = QTextEdit()
        self.transcript_viewer.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_viewer)
        
        layout.addWidget(transcript_group)
        
        # Prediction controls
        predict_group = QGroupBox("Prediction")
        predict_layout = QHBoxLayout(predict_group)
        
        predict_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spinner = QDoubleSpinBox()
        self.threshold_spinner.setRange(0.01, 0.99)
        self.threshold_spinner.setValue(0.5)
        self.threshold_spinner.setDecimals(2)
        self.threshold_spinner.setSingleStep(0.05)
        predict_layout.addWidget(self.threshold_spinner)
        
        self.predict_btn = QPushButton("Predict Silences")
        self.predict_btn.clicked.connect(self.predict_silences)
        self.predict_btn.setEnabled(False)
        predict_layout.addWidget(self.predict_btn)
        
        self.save_output_btn = QPushButton("Save Output")
        self.save_output_btn.clicked.connect(self.save_output)
        self.save_output_btn.setEnabled(False)
        predict_layout.addWidget(self.save_output_btn)
        
        layout.addWidget(predict_group)
        
        # Result viewer
        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout(result_group)
        
        self.result_viewer = QTextEdit()
        self.result_viewer.setReadOnly(True)
        result_layout.addWidget(self.result_viewer)
        
        layout.addWidget(result_group)
    
    def browse_training_data(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if dir_path:
            self.training_data_dir = dir_path
            self.training_data_path.setText(dir_path)
            
            # Check if there are transcript files in the directory
            txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
            if txt_files:
                self.data_status_label.setText(f"Found {len(txt_files)} transcript file(s)")
                self.train_btn.setEnabled(True)
            else:
                self.data_status_label.setText("No transcript files found in directory")
                self.train_btn.setEnabled(False)
    
    def train_model(self):
        # Load training data
        try:
            # Clear log viewer
            self.log_viewer.clear()
            self.add_log_message("Starting training process...")
            
            # Get LSTM parameters
            sequence_length = self.seq_length_spinner.value()
            stride = self.stride_spinner.value()
            
            # Update predictor with sequence length
            self.predictor = SilencePredictor(sequence_length=sequence_length)
            
            # Load sequences for LSTM training
            self.add_log_message(f"Loading training data from {self.training_data_dir}...")
            features, labels = TranscriptProcessor.load_training_sequences(
                self.training_data_dir, 
                sequence_length=sequence_length,
                stride=stride
            )
            
            if len(features) == 0:
                QMessageBox.warning(self, "Warning", "No valid training data found.")
                return
                
            self.data_status_label.setText(f"Loaded {len(features)} training sequences")
            self.add_log_message(f"Successfully loaded {len(features)} training sequences with shape {features.shape}")
            
            # Setup training thread
            epochs = self.epochs_spinner.value()
            batch_size = self.batch_size_spinner.value()
            
            self.add_log_message(f"Training with {epochs} epochs, batch size {batch_size}")
            self.training_thread = SilenceTrainerThread(
                self.predictor, features, labels, epochs, batch_size, sequence_mode=True
            )
            
            # Connect signals
            self.training_thread.progress_update.connect(self.update_training_progress)
            self.training_thread.training_finished.connect(self.on_training_finished)
            
            # Update UI
            self.train_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.loss_label.setText("Loss: N/A")
            
            # Start training
            self.add_log_message("Starting training thread...")
            self.training_thread.start()
            
        except Exception as e:
            import traceback
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            self.add_log_message(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", f"Training error: {str(e)}")
            self.data_status_label.setText(f"Error: {str(e)}")
    
    def add_log_message(self, message):
        """Add a message to the log viewer with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_viewer.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Process events to update UI
        QApplication.processEvents()
    
    def update_training_progress(self, progress, loss, message):
        self.progress_bar.setValue(progress)
        self.loss_label.setText(f"Loss: {loss:.6f}")
        
        # Add log message
        if message:
            self.add_log_message(message)
        
        # Redraw loss plot every 10% of progress or when explicitly requested
        if progress % 5 == 0 or "completed" in message.lower():
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.training_thread.losses, color='skyblue')
            ax.set_title("Training Loss")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            self.canvas.draw()
            
            # Process events to update UI
            QApplication.processEvents()
    
    def on_training_finished(self, losses):
        self.progress_bar.setValue(100)
        
        # Final plot update
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(losses, color='skyblue')
        ax.set_title("Training Loss")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        self.canvas.draw()
        
        # Update UI
        self.train_btn.setEnabled(True)
        self.save_model_btn.setEnabled(True)
        
        self.add_log_message("Training completed!")
        QMessageBox.information(self, "Success", "Model training completed!")
    
    def save_model(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "models/silence_model.pt", "PyTorch Model (*.pt)"
        )
        
        if file_path:
            try:
                model_dir = os.path.dirname(file_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    
                self.add_log_message(f"Saving model to {file_path}...")
                self.predictor.save_model(file_path)
                self.model_path = file_path
                
                self.add_log_message(f"Model saved successfully.")
                QMessageBox.information(self, "Success", f"Model saved to {file_path}")
            except Exception as e:
                self.add_log_message(f"Error saving model: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "models/", "PyTorch Model (*.pt)"
        )
        
        if file_path:
            try:
                self.add_log_message(f"Loading model from {file_path}...")
                self.predictor.load_model(file_path)
                self.model_path = file_path
                
                # Enable prediction
                self.predict_btn.setEnabled(True if self.transcript_path.text() else False)
                
                self.add_log_message(f"Model loaded successfully.")
                QMessageBox.information(self, "Success", f"Model loaded from {file_path}")
            except Exception as e:
                self.add_log_message(f"Error loading model: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def browse_transcript(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Transcript", "", "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                # Load and display transcript
                self.add_log_message(f"Loading transcript from {file_path}...")
                self.current_transcript = TranscriptProcessor.parse_transcript(file_path)
                self.transcript_path.setText(file_path)
                
                # Display in viewer
                self.display_transcript(self.current_transcript)
                
                # Enable prediction if model is loaded
                self.predict_btn.setEnabled(True)
                self.add_log_message(f"Transcript loaded with {len(self.current_transcript.entries)} entries.")
                
            except Exception as e:
                self.add_log_message(f"Error loading transcript: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to load transcript: {str(e)}")
    
    def display_transcript(self, transcript):
        text = ""
        for entry in transcript.entries:
            text += f"[{entry.start_time:.2f} -> {entry.end_time:.2f}] {entry.text}\n"
            
            # If not the last entry, show silence gap
            if entry != transcript.entries[-1]:
                next_entry = transcript.entries[transcript.entries.index(entry) + 1]
                silence_duration = next_entry.start_time - entry.end_time
                if silence_duration > 0.01:
                    text += f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] \n"
        
        self.transcript_viewer.setText(text)
    
    def predict_silences(self):
        if not self.current_transcript:
            return
        
        try:
            # Extract features from transcript
            self.add_log_message("Extracting features from transcript...")
            features = self.current_transcript.to_features()
            
            if len(features) == 0:
                self.add_log_message("No valid features found in transcript.")
                QMessageBox.warning(self, "Warning", "No valid features found in transcript.")
                return
            
            self.add_log_message(f"Extracted {len(features)} feature vectors.")
            
            # Get prediction threshold
            threshold = self.threshold_spinner.value()
            
            # Make predictions (True = keep, False = cut)
            self.add_log_message(f"Making predictions with threshold {threshold}...")
            keep_predictions = self.predictor.predict(features, threshold)
            self.cut_predictions = [not p for p in keep_predictions]  # Invert to get cut markers
            
            # Count cut markers
            cut_count = sum(self.cut_predictions)
            total_count = len(self.cut_predictions)
            self.add_log_message(f"Prediction complete. Marked {cut_count}/{total_count} silences for cutting.")
            
            # Display result
            self.display_result()
            
            # Enable save button
            self.save_output_btn.setEnabled(True)
            
        except Exception as e:
            import traceback
            error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
            self.add_log_message(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", f"Prediction error: {str(e)}")
    
    def display_result(self):
        if not self.current_transcript or not self.cut_predictions:
            return
        
        text = ""
        cut_idx = 0
        
        for i, entry in enumerate(self.current_transcript.entries):
            text += f"[{entry.start_time:.2f} -> {entry.end_time:.2f}] {entry.text}\n"
            
            # If not the last entry, show silence gap
            if i < len(self.current_transcript.entries) - 1:
                next_entry = self.current_transcript.entries[i + 1]
                silence_duration = next_entry.start_time - entry.end_time
                
                if silence_duration > 0.01:
                    if cut_idx < len(self.cut_predictions) and self.cut_predictions[cut_idx]:
                        text += f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] [SILENCE-CUT]\n"
                    else:
                        text += f"[{entry.end_time:.2f} -> {next_entry.start_time:.2f}] \n"
                    
                    cut_idx += 1
        
        self.result_viewer.setText(text)
    
    def save_output(self):
        if not self.current_transcript or not self.cut_predictions:
            return
        
        # Get original file path
        original_path = self.transcript_path.text()
        
        # Create default output path
        file_name = os.path.basename(original_path)
        base_name, ext = os.path.splitext(file_name)
        default_output = os.path.join(os.path.dirname(original_path), f"{base_name}_cleaned{ext}")
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Transcript", default_output, "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                # Save the processed transcript
                self.add_log_message(f"Saving processed transcript to {file_path}...")
                TranscriptProcessor.save_processed_transcript(
                    self.current_transcript, file_path, self.cut_predictions
                )
                
                self.add_log_message("Transcript saved successfully.")
                QMessageBox.information(self, "Success", f"Processed transcript saved to {file_path}")
            except Exception as e:
                self.add_log_message(f"Error saving transcript: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to save transcript: {str(e)}")

def main():
    # Create application directory structure
    os.makedirs("models", exist_ok=True)
    
    # Start application
    app = QApplication(sys.argv)
    window = PytorchSilencerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 