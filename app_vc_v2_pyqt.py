import sys
import os
import torch
import yaml
import tempfile
import numpy as np
import librosa
import soundfile as sf
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Generator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QSlider, QCheckBox,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QProgressBar, QGroupBox,
    QFormLayout, QComboBox, QStyleFactory, QMessageBox, QSpacerItem, QSizePolicy,
    QListWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QUrl, QBuffer, QIODevice, QByteArray
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaFormat
from PyQt6.QtGui import QIcon, QFont, QAction
from pydub import AudioSegment

# Determine device for PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

# Media file extensions and formats
AUDIO_EXTENSIONS = "Audio Files (*.wav *.mp3 *.flac *.ogg)"

# Description for the application window
APP_DESCRIPTION = """
<b>Seed Voice Conversion V2</b> - Zero-shot voice conversion with in-context learning
<br>Reference audio will be clipped to 25s if longer. Source audio will be processed in chunks if total exceeds 30s.
<br>Bulk processing enabled: Add multiple source files to process with a single reference audio.
<br>Visit <a href="https://github.com/Plachtaa/seed-vc">GitHub repository</a> for more details.
"""

def load_models(args):
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper


class VoiceConversionThread(QThread):
    progressUpdate = pyqtSignal(int)  # Signal for progress updates (0-100)
    statusUpdate = pyqtSignal(str)    # Signal for status text updates
    streamOutput = pyqtSignal(bytes)  # Signal for streaming audio data
    conversionComplete = pyqtSignal(str)  # Signal with path to output file
    errorOccurred = pyqtSignal(str)   # Signal for error reporting

    def __init__(self, vc_wrapper, source_path, target_path, params, output_dir):
        super().__init__()
        self.vc_wrapper = vc_wrapper
        self.source_path = source_path
        self.target_path = target_path
        self.params = params
        self.output_dir = output_dir
        self.cancel_requested = False
        
        # Create output filename based on source filename with timestamp
        source_basename = os.path.basename(source_path)
        source_name = os.path.splitext(source_basename)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.output_dir, f"{source_name}_{timestamp}.wav")
        
    def run(self):
        try:
            self.statusUpdate.emit("Starting voice conversion...")
            
            # Extract parameters from params dictionary
            diffusion_steps = self.params.get('diffusion_steps', 30)
            length_adjust = self.params.get('length_adjust', 1.0)
            intelligibility_cfg_rate = self.params.get('intelligibility_cfg_rate', 0.5)
            similarity_cfg_rate = self.params.get('similarity_cfg_rate', 0.5)
            top_p = self.params.get('top_p', 0.9)
            temperature = self.params.get('temperature', 1.0)
            repetition_penalty = self.params.get('repetition_penalty', 1.0)
            convert_style = self.params.get('convert_style', False)
            anonymization_only = self.params.get('anonymization_only', False)
            
            self.statusUpdate.emit(f"Processing with {diffusion_steps} diffusion steps...")
            
            # Generator that yields streaming chunks
            stream_generator = self.vc_wrapper.convert_voice_with_streaming(
                source_audio_path=self.source_path,
                target_audio_path=self.target_path,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                intelligebility_cfg_rate=intelligibility_cfg_rate,
                similarity_cfg_rate=similarity_cfg_rate,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                convert_style=convert_style,
                anonymization_only=anonymization_only,
                device=device,
                dtype=dtype,
                stream_output=True
            )
            
            # Process streaming chunks
            final_audio = None
            chunk_count = 0
            total_chunks_estimate = 10  # Just an initial estimate, will be refined
            
            for chunk_data, full_output in stream_generator:
                if self.cancel_requested:
                    self.statusUpdate.emit("Conversion canceled.")
                    return
                
                # Emit the streaming data
                self.streamOutput.emit(chunk_data)
                
                # Keep track of the final audio when available
                if full_output is not None:
                    final_audio = full_output
                
                # Update progress
                chunk_count += 1
                progress = min(int(chunk_count / total_chunks_estimate * 100), 99)
                self.progressUpdate.emit(progress)
                self.statusUpdate.emit(f"Processing chunk {chunk_count}...")
            
            # Save the complete audio to the output file
            if final_audio is not None:
                sr, audio_data = final_audio
                sf.write(self.output_path, audio_data, sr)
                self.statusUpdate.emit("Conversion complete!")
                self.progressUpdate.emit(100)
                self.conversionComplete.emit(self.output_path)
            else:
                self.errorOccurred.emit("No output audio was generated.")
                
        except Exception as e:
            self.errorOccurred.emit(f"Error during conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def cancel(self):
        self.cancel_requested = True


class AudioPlayer:
    """Helper class to handle audio playback."""
    
    def __init__(self, parent=None):
        self.media_player = QMediaPlayer(parent)
        self.audio_output = QAudioOutput(parent)
        self.media_player.setAudioOutput(self.audio_output)
        
        # Connect signals
        self.media_player.errorOccurred.connect(self._handle_error)
        
    def play_file(self, file_path):
        """Play audio from a file path."""
        if not file_path or not os.path.exists(file_path):
            return False
        
        url = QUrl.fromLocalFile(file_path)
        self.media_player.setSource(url)
        self.media_player.play()
        return True
        
    def play_data(self, data):
        """Play audio from bytes data."""
        # TODO: Implement buffer handling for byte data
        pass
        
    def stop(self):
        """Stop playback."""
        self.media_player.stop()
        
    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)."""
        self.audio_output.setVolume(volume)
        
    def _handle_error(self, error, error_string):
        if error != QMediaPlayer.Error.NoError:
            print(f"Media player error: {error_string}")


class SliderWithValue(QWidget):
    """Custom slider widget with a value label."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, 
                 min_value, 
                 max_value, 
                 default_value, 
                 step=1, 
                 label="", 
                 info="",
                 parent=None):
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.scale_factor = 1.0
        
        # If step is decimal, we need to scale for the QSlider
        if isinstance(step, float):
            if step < 1.0:
                decimal_places = len(str(step).split('.')[-1])
                self.scale_factor = 10 ** decimal_places
        
        # Create layouts
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add label and info if provided
        if label:
            header_layout = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            header_layout.addWidget(label_widget)
            
            if info:
                info_label = QLabel(f"({info})")
                info_label.setStyleSheet("color: gray;")
                header_layout.addWidget(info_label)
                
            header_layout.addStretch(1)
            layout.addLayout(header_layout)
        
        # Create slider controls layout
        slider_layout = QHBoxLayout()
        
        # Create the slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_value * self.scale_factor))
        self.slider.setMaximum(int(max_value * self.scale_factor))
        self.slider.setValue(int(default_value * self.scale_factor))
        self.slider.setSingleStep(int(step * self.scale_factor))
        
        # Create value display
        self.value_label = QLabel(str(default_value))
        self.value_label.setMinimumWidth(40)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        # Add widgets to layout
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.value_label)
        layout.addLayout(slider_layout)
        
        # Connect signals
        self.slider.valueChanged.connect(self._update_value)
        
    def _update_value(self, integer_value):
        actual_value = integer_value / self.scale_factor
        self.value_label.setText(f"{actual_value:.1f}" if self.scale_factor > 1 else f"{actual_value}")
        self.valueChanged.emit(actual_value)
        
    def value(self):
        """Get the current slider value."""
        return self.slider.value() / self.scale_factor
    
    def setValue(self, value):
        """Set the slider value."""
        self.slider.setValue(int(value * self.scale_factor))


class MainWindow(QMainWindow):
    def __init__(self, vc_wrapper, output_dir, examples=None):
        super().__init__()
        self.vc_wrapper = vc_wrapper
        self.output_dir = output_dir
        self.examples = examples or []
        self.audio_players = {
            'source': AudioPlayer(self),
            'target': AudioPlayer(self),
            'output': AudioPlayer(self)
        }
        self.conversion_thread = None
        self.output_file_path = None
        self.source_files = []  # List to store source file paths
        self.current_file_index = -1  # Index of currently processing file (-1 = not processing)
        self.bulk_processing = False  # Flag to indicate bulk processing is active
        
        self.initUI()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle("Seed Voice Conversion V2")
        self.setMinimumSize(800, 600)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Add description label
        desc_label = QLabel(APP_DESCRIPTION)
        desc_label.setOpenExternalLinks(True)
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)

        # Add output folder display
        output_folder_layout = QHBoxLayout()
        output_folder_label = QLabel("Output Folder:")
        self.output_folder_path = QLabel(self.output_dir)
        self.output_folder_path.setStyleSheet("font-weight: bold;")
        output_folder_open_btn = QPushButton("Open Folder")
        output_folder_open_btn.clicked.connect(self.openOutputFolder)
        
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_path, 1)
        output_folder_layout.addWidget(output_folder_open_btn)
        main_layout.addLayout(output_folder_layout)
        
        # File input section
        file_group = QGroupBox("Audio Input Files")
        file_layout = QVBoxLayout(file_group)
        
        # Source audio selection
        source_layout = QVBoxLayout()  # Changed to vertical layout for list
        source_header = QHBoxLayout()
        source_label = QLabel("Source Audio Files:")
        source_header.addWidget(source_label)
        source_header.addStretch(1)
        source_layout.addLayout(source_header)
        
        # Add list widget to display selected files
        self.source_files_list = QListWidget()
        self.source_files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.source_files_list.setMinimumHeight(100)  # Give some vertical space
        source_layout.addWidget(self.source_files_list)
        
        # Add buttons for file management
        source_buttons = QHBoxLayout()
        source_add_btn = QPushButton("Add Files...")
        source_add_btn.clicked.connect(self.addSourceFiles)
        source_remove_btn = QPushButton("Remove Selected")
        source_remove_btn.clicked.connect(self.removeSourceFiles)
        source_play_btn = QPushButton("Play Selected")
        source_play_btn.clicked.connect(lambda: self.playSelectedSource())
        
        source_buttons.addWidget(source_add_btn)
        source_buttons.addWidget(source_remove_btn)
        source_buttons.addWidget(source_play_btn)
        source_buttons.addStretch(1)
        source_layout.addLayout(source_buttons)
        file_layout.addLayout(source_layout)
        
        # Target audio selection
        target_layout = QHBoxLayout()
        target_label = QLabel("Reference Audio:")
        self.target_path_label = QLabel("No file selected")
        self.target_path_label.setStyleSheet("color: gray;")
        target_browse_btn = QPushButton("Browse...")
        target_browse_btn.clicked.connect(self.selectTargetFile)
        target_play_btn = QPushButton("Play")
        target_play_btn.clicked.connect(lambda: self.playAudio('target'))
        
        target_layout.addWidget(target_label)
        target_layout.addWidget(self.target_path_label, 1)
        target_layout.addWidget(target_browse_btn)
        target_layout.addWidget(target_play_btn)
        file_layout.addLayout(target_layout)
        
        main_layout.addWidget(file_group)
        
        # Parameters section
        params_group = QGroupBox("Conversion Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Create sliders
        self.diffusion_steps_slider = SliderWithValue(
            min_value=1, max_value=200, default_value=30, step=1,
            label="Diffusion Steps", 
            info="30 by default, 50~100 for best quality"
        )
        params_layout.addWidget(self.diffusion_steps_slider)
        
        self.length_adjust_slider = SliderWithValue(
            min_value=0.5, max_value=2.0, default_value=1.0, step=0.1,
            label="Length Adjust", 
            info="<1.0 speeds up speech, >1.0 slows down speech"
        )
        params_layout.addWidget(self.length_adjust_slider)
        
        self.intelligibility_slider = SliderWithValue(
            min_value=0.0, max_value=1.0, default_value=0.5, step=0.1,
            label="Intelligibility CFG Rate", 
            info="Has subtle influence on clarity"
        )
        params_layout.addWidget(self.intelligibility_slider)
        
        self.similarity_slider = SliderWithValue(
            min_value=0.0, max_value=1.0, default_value=0.5, step=0.1,
            label="Similarity CFG Rate", 
            info="Has subtle influence on voice similarity"
        )
        params_layout.addWidget(self.similarity_slider)
        
        self.top_p_slider = SliderWithValue(
            min_value=0.1, max_value=1.0, default_value=0.9, step=0.1,
            label="Top-p", 
            info="Controls diversity of generated audio"
        )
        params_layout.addWidget(self.top_p_slider)
        
        self.temperature_slider = SliderWithValue(
            min_value=0.1, max_value=2.0, default_value=1.0, step=0.1,
            label="Temperature", 
            info="Controls randomness of generated audio"
        )
        params_layout.addWidget(self.temperature_slider)
        
        self.repetition_penalty_slider = SliderWithValue(
            min_value=1.0, max_value=3.0, default_value=1.0, step=0.1,
            label="Repetition Penalty", 
            info="Penalizes repetition in generated audio"
        )
        params_layout.addWidget(self.repetition_penalty_slider)
        
        # Checkboxes
        checkbox_layout = QHBoxLayout()
        self.convert_style_checkbox = QCheckBox("Convert Style")
        self.anonymization_checkbox = QCheckBox("Anonymization Only")
        checkbox_layout.addWidget(self.convert_style_checkbox)
        checkbox_layout.addWidget(self.anonymization_checkbox)
        checkbox_layout.addStretch(1)
        params_layout.addLayout(checkbox_layout)
        
        main_layout.addWidget(params_group)
        
        # Example section (if available)
        if self.examples:
            examples_group = QGroupBox("Examples")
            examples_layout = QHBoxLayout(examples_group)
            
            for i, example in enumerate(self.examples):
                example_btn = QPushButton(f"Example {i+1}")
                example_btn.clicked.connect(lambda checked, idx=i: self.loadExample(idx))
                examples_layout.addWidget(example_btn)
                
            examples_layout.addStretch(1)
            main_layout.addWidget(examples_group)
        
        # Conversion control section
        control_group = QGroupBox("Conversion Control")
        control_layout = QVBoxLayout(control_group)
        
        # Add convert button
        button_layout = QHBoxLayout()
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.clicked.connect(self.startConversion)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancelConversion)
        
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.cancel_btn)
        control_layout.addLayout(button_layout)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_group)
        
        # Output section
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        
        # Add output controls
        self.output_play_btn = QPushButton("Play Output")
        self.output_play_btn.setEnabled(False)
        self.output_play_btn.clicked.connect(lambda: self.playAudio('output'))
        
        self.output_stop_btn = QPushButton("Stop")
        self.output_stop_btn.setEnabled(False)
        self.output_stop_btn.clicked.connect(lambda: self.stopAudio('output'))
        
        self.last_output_label = QLabel("No output generated yet")
        self.last_output_label.setStyleSheet("color: gray;")
        
        output_layout.addWidget(self.output_play_btn)
        output_layout.addWidget(self.output_stop_btn)
        output_layout.addWidget(self.last_output_label, 1)
        
        main_layout.addWidget(output_group)
        
        # Add some stretch at the end
        main_layout.addStretch(1)
        
    def openOutputFolder(self):
        """Open the output folder in the file explorer."""
        if os.path.exists(self.output_dir):
            # Use the appropriate command based on the operating system
            if sys.platform == 'win32':
                os.startfile(self.output_dir)
            elif sys.platform == 'darwin':  # macOS
                import subprocess
                subprocess.Popen(['open', self.output_dir])
            else:  # Linux and other Unix-like
                import subprocess
                subprocess.Popen(['xdg-open', self.output_dir])
                
    def addSourceFiles(self):
        """Open a file dialog to select multiple source audio files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Source Audio Files", "", AUDIO_EXTENSIONS
        )
        if file_paths:
            for path in file_paths:
                if path not in self.source_files:  # Avoid duplicates
                    self.source_files.append(path)
                    self.source_files_list.addItem(os.path.basename(path))
    
    def removeSourceFiles(self):
        """Remove selected files from the source files list."""
        selected_items = self.source_files_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            row = self.source_files_list.row(item)
            self.source_files_list.takeItem(row)
            del self.source_files[row]
            
    def playSelectedSource(self):
        """Play the selected source audio file."""
        selected_items = self.source_files_list.selectedItems()
        if not selected_items:
            return
            
        # Just play the first selected file
        item = selected_items[0]
        row = self.source_files_list.row(item)
        file_path = self.source_files[row]
        self.audio_players['source'].play_file(file_path)
            
    def selectTargetFile(self):
        """Open a file dialog to select the target reference audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio File", "", AUDIO_EXTENSIONS
        )
        if file_path:
            self.target_path_label.setText(file_path)
            self.target_path_label.setStyleSheet("")
            
    def playAudio(self, audio_type):
        """Play the selected audio file."""
        if audio_type == 'source':
            # For source files, use playSelectedSource instead
            self.playSelectedSource()
        elif audio_type == 'target':
            file_path = self.target_path_label.text()
            if file_path != "No file selected":
                self.audio_players['target'].play_file(file_path)
        elif audio_type == 'output' and self.output_file_path:
            self.audio_players['output'].play_file(self.output_file_path)
            
    def stopAudio(self, audio_type):
        """Stop audio playback."""
        self.audio_players[audio_type].stop()
            
    def loadExample(self, example_idx):
        """Load a predefined example."""
        if example_idx < len(self.examples):
            example = self.examples[example_idx]
            
            # Clear previous files and add the example source file
            self.source_files = [example[0]]
            self.source_files_list.clear()
            self.source_files_list.addItem(os.path.basename(example[0]))
            
            self.target_path_label.setText(example[1])
            self.target_path_label.setStyleSheet("")
            
            # Set parameters
            self.diffusion_steps_slider.setValue(example[2])
            self.length_adjust_slider.setValue(example[3])
            self.intelligibility_slider.setValue(example[4])
            self.similarity_slider.setValue(example[5])
            self.top_p_slider.setValue(example[6])
            self.temperature_slider.setValue(example[7])
            self.repetition_penalty_slider.setValue(example[8])
            
            # Set checkboxes
            self.convert_style_checkbox.setChecked(example[9])
            self.anonymization_checkbox.setChecked(example[10])
            
    def startConversion(self):
        """Start the voice conversion process for all source files."""
        if not self.source_files:
            QMessageBox.warning(self, "Input Required", 
                               "Please add at least one source audio file.")
            return
            
        target_path = self.target_path_label.text()
        if target_path == "No file selected":
            QMessageBox.warning(self, "Input Required", 
                               "Please select a reference audio file.")
            return
                
        # Disable controls during conversion
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Get parameters
        params = {
            'diffusion_steps': int(self.diffusion_steps_slider.value()),
            'length_adjust': self.length_adjust_slider.value(),
            'intelligibility_cfg_rate': self.intelligibility_slider.value(),
            'similarity_cfg_rate': self.similarity_slider.value(),
            'top_p': self.top_p_slider.value(),
            'temperature': self.temperature_slider.value(),
            'repetition_penalty': self.repetition_penalty_slider.value(),
            'convert_style': self.convert_style_checkbox.isChecked(),
            'anonymization_only': self.anonymization_checkbox.isChecked()
        }
        
        # Start bulk processing
        self.bulk_processing = True
        self.current_file_index = -1
        self.updateStatus(f"Starting bulk processing of {len(self.source_files)} files...")
        
        # Process the first file
        self.processNextFile(params, target_path)
        
    def processNextFile(self, params, target_path):
        """Process the next file in the queue."""
        self.current_file_index += 1
        
        if self.current_file_index >= len(self.source_files):
            # We're done with all files
            self.bulk_processing = False
            self.current_file_index = -1
            self.updateStatus(f"Bulk processing complete! All {len(self.source_files)} files processed.")
            self.convert_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            return
        
        # Get the current file to process
        current_source = self.source_files[self.current_file_index]
        filename = os.path.basename(current_source)
        
        # Update UI to show current progress
        self.updateStatus(f"Processing file {self.current_file_index + 1} of {len(self.source_files)}: {filename}")
        self.source_files_list.setCurrentRow(self.current_file_index)
        
        # Create and start the conversion thread
        self.conversion_thread = VoiceConversionThread(
            self.vc_wrapper, current_source, target_path, params, self.output_dir
        )
        
        # Connect signals
        self.conversion_thread.progressUpdate.connect(self.updateProgress)
        self.conversion_thread.statusUpdate.connect(self.updateStatus)
        self.conversion_thread.streamOutput.connect(self.handleStreamOutput)
        self.conversion_thread.conversionComplete.connect(self.handleConversionComplete)
        self.conversion_thread.errorOccurred.connect(self.handleConversionError)
        self.conversion_thread.finished.connect(self.handleThreadFinished)
        
        # Start the thread
        self.conversion_thread.start()
            
    def cancelConversion(self):
        """Cancel the ongoing conversion process."""
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.updateStatus("Cancelling conversion...")
            self.conversion_thread.cancel()
        
        # Reset bulk processing status
        self.bulk_processing = False
        self.current_file_index = -1
            
    def updateProgress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
        
    def updateStatus(self, status_text):
        """Update the status label."""
        self.status_label.setText(status_text)
        
    def handleStreamOutput(self, audio_data):
        """Handle streaming audio output."""
        # Just update the status for now
        self.updateStatus("Received audio stream chunk...")
        
    def handleConversionComplete(self, output_path):
        """Handle completion of the conversion process."""
        self.output_file_path = output_path
        filename = os.path.basename(output_path)
        self.last_output_label.setText(filename)
        self.last_output_label.setStyleSheet("")
        
        if self.bulk_processing:
            self.updateStatus(f"File {self.current_file_index + 1} of {len(self.source_files)} complete! Output saved to: {filename}")
        else:
            self.updateStatus(f"Conversion complete! Output saved to: {filename}")
        
        # Enable output controls
        self.output_play_btn.setEnabled(True)
        self.output_stop_btn.setEnabled(True)
        
        # Play the output
        self.playAudio('output')
        
        # If we're in bulk processing mode, move to the next file
        if self.bulk_processing:
            # Get parameters again (in case they changed)
            params = {
                'diffusion_steps': int(self.diffusion_steps_slider.value()),
                'length_adjust': self.length_adjust_slider.value(),
                'intelligibility_cfg_rate': self.intelligibility_slider.value(),
                'similarity_cfg_rate': self.similarity_slider.value(),
                'top_p': self.top_p_slider.value(),
                'temperature': self.temperature_slider.value(),
                'repetition_penalty': self.repetition_penalty_slider.value(),
                'convert_style': self.convert_style_checkbox.isChecked(),
                'anonymization_only': self.anonymization_checkbox.isChecked()
            }
            target_path = self.target_path_label.text()
            self.processNextFile(params, target_path)
        
    def handleConversionError(self, error_message):
        """Handle errors during conversion."""
        if self.bulk_processing:
            current_file = os.path.basename(self.source_files[self.current_file_index])
            QMessageBox.critical(self, "Conversion Error", 
                              f"Error processing file {self.current_file_index + 1} ({current_file}): {error_message}")
            self.updateStatus(f"Error on file {self.current_file_index + 1}: {error_message}")
            
            # If in bulk processing, try to continue with next file
            params = {
                'diffusion_steps': int(self.diffusion_steps_slider.value()),
                'length_adjust': self.length_adjust_slider.value(),
                'intelligibility_cfg_rate': self.intelligibility_slider.value(),
                'similarity_cfg_rate': self.similarity_slider.value(),
                'top_p': self.top_p_slider.value(),
                'temperature': self.temperature_slider.value(),
                'repetition_penalty': self.repetition_penalty_slider.value(),
                'convert_style': self.convert_style_checkbox.isChecked(),
                'anonymization_only': self.anonymization_checkbox.isChecked()
            }
            target_path = self.target_path_label.text()
            self.processNextFile(params, target_path)
        else:
            QMessageBox.critical(self, "Conversion Error", error_message)
            self.updateStatus(f"Error: {error_message}")
        
    def handleThreadFinished(self):
        """Handle thread completion."""
        # Re-enable controls
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any ongoing conversion
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.conversion_thread.cancel()
            self.conversion_thread.wait(1000)  # Wait up to 1 second for thread to finish
            
        # Stop any audio playback
        for player in self.audio_players.values():
            player.stop()
            
        # Accept the close event
        event.accept()


def main(args):
    # Create a standard output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load models
    vc_wrapper = load_models(args)
    
    # Define examples
    examples = [
        ["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
        ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
    ]
    
    # Create and show the application
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))  # Use Fusion style for better appearance
    window = MainWindow(vc_wrapper, output_dir=output_dir, examples=examples)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save converted audio files")
    args = parser.parse_args()
    main(args)
