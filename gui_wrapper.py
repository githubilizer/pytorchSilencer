import sys
import os
import subprocess
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QTextEdit, QTabWidget, 
                            QSpinBox, QDoubleSpinBox, QProgressBar,
                            QGroupBox, QMessageBox, QLineEdit, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont

class CommandThread(QThread):
    progress_update = pyqtSignal(str)
    command_finished = pyqtSignal(bool, str)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
        
    def run(self):
        try:
            # Print the command being run for transparency
            command_str = " ".join(self.command)
            self.progress_update.emit(f"Running command: {command_str}")
            
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Remove any carriage returns and trim whitespace
                    clean_line = line.replace('\r', '').strip()
                    if clean_line:
                        self.progress_update.emit(clean_line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                self.command_finished.emit(True, "Command completed successfully")
            else:
                self.command_finished.emit(False, f"Command failed with return code {return_code}")
                
        except Exception as e:
            self.command_finished.emit(False, f"Error executing command: {str(e)}")


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

        # Initialize paths
        self.training_data_dir = ""

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

        if self.input_transcript:
            base, ext = os.path.splitext(self.input_transcript)
            hhmm = datetime.datetime.now().strftime("%I%M%p").lower()
            self.output_transcript = f"{base}_processed_{hhmm}{ext}"
        else:
            self.output_transcript = ""

        # Default paths for Audio Silencer tab
        self.processed_transcript_path = self.output_transcript
        self.input_video_path = ""
        self.output_video_path = ""

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
        data_layout = QHBoxLayout(data_group)
        
        self.training_data_path = QLineEdit()
        self.training_data_path.setReadOnly(True)
        self.training_data_path.setPlaceholderText("Select training data directory...")
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_training_data)
        
        # Add quick load button for GOODFULLSCRIPTSDATA
        quick_load_button = QPushButton("Load Full Scripts")
        quick_load_button.setToolTip("Load training data from GOODFULLSCRIPTSDATA folder")
        quick_load_button.clicked.connect(self.load_full_scripts_data)
        
        data_layout.addWidget(self.training_data_path)
        data_layout.addWidget(browse_button)
        data_layout.addWidget(quick_load_button)
        
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
        
    def browse_training_data(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if directory:
            self.training_data_dir = directory
            self.training_data_path.setText(directory)
            
            # Check for transcript files
            count = sum(1 for f in os.listdir(directory) if f.endswith('.txt'))
            self.data_status_label.setText(f"Found {count} transcript files")
            
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
            
    def train_model(self):
        # Validate inputs
        if not self.training_data_dir:
            QMessageBox.warning(self, "No Data", "Please select a training data directory first.")
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
        
        # Build command
        command = [
            "python", "lstm_demo.py", "train",
            "--data", self.training_data_dir,
            "--model", model_path,
            "--seq-length", str(seq_length),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size)
        ]
        
        # Run command in thread
        self.command_thread = CommandThread(command)
        self.command_thread.progress_update.connect(self.add_log_message)
        self.command_thread.command_finished.connect(self.on_training_finished)
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

    def browse_audio_transcript(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Processed Transcript", "", "Text Files (*.txt)"
        )
        if file_path:
            self.processed_transcript_path = file_path
            self.audio_transcript_field.setText(file_path)
            if not self.output_video_path:
                base, _ = os.path.splitext(file_path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_video_path = f"{base}_cut_{timestamp}.mp4"
                self.output_video_field.setText(self.output_video_path)

    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.mov *.mkv)"
        )
        if file_path:
            self.input_video_path = file_path
            self.video_path_field.setText(file_path)

    def browse_output_video(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Video", "", "Video Files (*.mp4 *.mov *.mkv)"
        )
        if file_path:
            base, ext = os.path.splitext(file_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_video_path = f"{base}_{timestamp}{ext}"
            self.output_video_field.setText(self.output_video_path)
            
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
        threshold = self.threshold_spinner.value()
        keep_ratio = self.keep_ratio_spinner.value()
        
        # Clear log
        self.process_log_viewer.clear()
        self.add_process_log("Starting transcript processing...")
        
        # Disable button
        self.process_btn.setEnabled(False)
        
        # Build command
        command = [
            "python", "lstm_demo.py", "process",
            "--model", model_path,
            "--input", input_path,
            "--output", output_path,
            "--threshold", str(threshold),
            "--keep-ratio", str(keep_ratio),
        ]
        
        # Run command in thread
        self.process_thread = CommandThread(command)
        self.process_thread.progress_update.connect(self.add_process_log)
        self.process_thread.command_finished.connect(self.on_processing_finished)
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
        base, ext = os.path.splitext(output_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_{timestamp}{ext}"
        self.output_video_path = output_path
        self.output_video_field.setText(output_path)

        self.audio_log_viewer.clear()
        self.add_audio_log("Starting video cutting...")
        self.audio_process_btn.setEnabled(False)

        command = [
            "python", "audio_silencer.py",
            "--video", video_path,
            "--transcript", transcript_path,
            "--output", output_path,
        ]

        self.video_thread = CommandThread(command)
        self.video_thread.progress_update.connect(self.add_audio_log)
        self.video_thread.command_finished.connect(self.on_video_finished)
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
        else:
            self.add_process_log(f"Processing failed: {message}")
            QMessageBox.critical(self, "Error", f"Processing failed: {message}")

    def load_full_scripts_data(self):
        """Hardcoded function to load data from GOODFULLSCRIPTSDATA folder"""
        data_path = "/home/j/Desktop/code/pytorchSilencer/GOODFULLSCRIPTSDATA"
        if os.path.exists(data_path) and os.path.isdir(data_path):
            self.training_data_dir = data_path
            self.training_data_path.setText(data_path)
            
            # Check for transcript files
            count = sum(1 for f in os.listdir(data_path) if f.endswith('.txt'))
            self.data_status_label.setText(f"Found {count} transcript files in Full Scripts data")
            self.add_log_message(f"Loaded Full Scripts data with {count} transcript files")
        else:
            QMessageBox.warning(self, "Error", "GOODFULLSCRIPTSDATA folder not found at expected location")

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
