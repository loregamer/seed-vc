import sys
import os
import torch
import yaml
import tempfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QSlider, QFileDialog, QCheckBox, 
                           QGroupBox, QProgressBar, QMessageBox, QSplitter, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QIcon, QFont

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str, str)  # stream_output_path, full_output_path
    error_signal = pyqtSignal(str)

    def __init__(self, vc_wrapper, params):
        super().__init__()
        self.vc_wrapper = vc_wrapper
        self.params = params

    def run(self):
        try:
            # Create temporary files for outputs
            stream_temp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            full_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            stream_temp.close()
            full_temp.close()
            
            stream_output_path = stream_temp.name
            full_output_path = full_temp.name
            
            # Extract parameters
            source_path = self.params['source_path']
            reference_path = self.params['reference_path']
            diffusion_steps = self.params['diffusion_steps']
            length_adjust = self.params['length_adjust']
            intelligibility_cfg = self.params['intelligibility_cfg']
            similarity_cfg = self.params['similarity_cfg']
            top_p = self.params['top_p']
            temperature = self.params['temperature']
            repetition_penalty = self.params['repetition_penalty']
            convert_style = self.params['convert_style']
            anonymization_only = self.params['anonymization_only']
            
            # Custom progress callback to update the progress bar
            def progress_callback(current_step, total_steps):
                progress_percentage = int((current_step / total_steps) * 100)
                self.progress_signal.emit(progress_percentage)
            
            # Call the voice conversion function with streaming
            # We need to modify the original function to accept a progress callback
            # and to save both streaming and full outputs to files
            result = self.vc_wrapper.convert_voice_with_streaming(
                source_path, reference_path, diffusion_steps, length_adjust,
                intelligibility_cfg, similarity_cfg, top_p, temperature,
                repetition_penalty, convert_style, anonymization_only,
                progress_callback=progress_callback,
                stream_output_path=stream_output_path,
                full_output_path=full_output_path
            )
            
            # Signal completion
            self.finished_signal.emit(stream_output_path, full_output_path)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class AudioPlayerWidget(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.audio_path = None
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_audio)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_audio)
        
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_audio_file)
        
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.file_button)
        
        layout.addLayout(controls_layout)
        
        # File path label
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        layout.addWidget(self.file_path_label)
        
        self.setLayout(layout)
        
    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)"
        )
        if file_path:
            self.set_audio_file(file_path)
    
    def set_audio_file(self, file_path):
        self.audio_path = file_path
        self.file_path_label.setText(os.path.basename(file_path))
        self.play_button.setEnabled(True)
        self.player.setSource(QUrl.fromLocalFile(file_path))
    
    def play_audio(self):
        if self.audio_path:
            self.player.play()
            self.stop_button.setEnabled(True)
    
    def stop_audio(self):
        self.player.stop()
        self.stop_button.setEnabled(False)
    
    def get_audio_path(self):
        return self.audio_path


class VoiceConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.vc_wrapper = None
        self.worker_thread = None
        self.stream_output_path = None
        self.full_output_path = None
        
        self.setWindowTitle("Seed Voice Conversion V2")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Description
        description_label = QLabel(
            "Zero-shot voice conversion with in-context learning. "
            "For local deployment please check GitHub repository for details and updates.\n"
            "Note that any reference audio will be forcefully clipped to 25s if beyond this length.\n"
            "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks."
        )
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)
        
        # Splitter for input/output sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Input section
        input_widget = QWidget()
        input_layout = QVBoxLayout()
        
        # Audio input section
        audio_inputs_layout = QHBoxLayout()
        
        # Source audio widget
        self.source_audio = AudioPlayerWidget("Source Audio")
        audio_inputs_layout.addWidget(self.source_audio)
        
        # Reference audio widget
        self.reference_audio = AudioPlayerWidget("Reference Audio")
        audio_inputs_layout.addWidget(self.reference_audio)
        
        input_layout.addLayout(audio_inputs_layout)
        
        # Parameters section
        params_group = QGroupBox("Conversion Parameters")
        params_layout = QGridLayout()
        
        # Diffusion Steps slider
        params_layout.addWidget(QLabel("Diffusion Steps:"), 0, 0)
        self.diffusion_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.diffusion_steps_slider.setMinimum(1)
        self.diffusion_steps_slider.setMaximum(200)
        self.diffusion_steps_slider.setValue(30)
        params_layout.addWidget(self.diffusion_steps_slider, 0, 1)
        self.diffusion_steps_value = QLabel("30")
        params_layout.addWidget(self.diffusion_steps_value, 0, 2)
        self.diffusion_steps_slider.valueChanged.connect(
            lambda v: self.diffusion_steps_value.setText(str(v))
        )
        
        # Length Adjust slider
        params_layout.addWidget(QLabel("Length Adjust:"), 1, 0)
        self.length_adjust_slider = QSlider(Qt.Orientation.Horizontal)
        self.length_adjust_slider.setMinimum(5)
        self.length_adjust_slider.setMaximum(20)
        self.length_adjust_slider.setValue(10)
        params_layout.addWidget(self.length_adjust_slider, 1, 1)
        self.length_adjust_value = QLabel("1.0")
        params_layout.addWidget(self.length_adjust_value, 1, 2)
        self.length_adjust_slider.valueChanged.connect(
            lambda v: self.length_adjust_value.setText(str(v/10))
        )
        
        # Intelligibility CFG Rate slider
        params_layout.addWidget(QLabel("Intelligibility CFG Rate:"), 2, 0)
        self.intelligibility_cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self.intelligibility_cfg_slider.setMinimum(0)
        self.intelligibility_cfg_slider.setMaximum(10)
        self.intelligibility_cfg_slider.setValue(5)
        params_layout.addWidget(self.intelligibility_cfg_slider, 2, 1)
        self.intelligibility_cfg_value = QLabel("0.5")
        params_layout.addWidget(self.intelligibility_cfg_value, 2, 2)
        self.intelligibility_cfg_slider.valueChanged.connect(
            lambda v: self.intelligibility_cfg_value.setText(str(v/10))
        )
        
        # Similarity CFG Rate slider
        params_layout.addWidget(QLabel("Similarity CFG Rate:"), 3, 0)
        self.similarity_cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_cfg_slider.setMinimum(0)
        self.similarity_cfg_slider.setMaximum(10)
        self.similarity_cfg_slider.setValue(5)
        params_layout.addWidget(self.similarity_cfg_slider, 3, 1)
        self.similarity_cfg_value = QLabel("0.5")
        params_layout.addWidget(self.similarity_cfg_value, 3, 2)
        self.similarity_cfg_slider.valueChanged.connect(
            lambda v: self.similarity_cfg_value.setText(str(v/10))
        )
        
        # Top-p slider
        params_layout.addWidget(QLabel("Top-p:"), 4, 0)
        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setMinimum(1)
        self.top_p_slider.setMaximum(10)
        self.top_p_slider.setValue(9)
        params_layout.addWidget(self.top_p_slider, 4, 1)
        self.top_p_value = QLabel("0.9")
        params_layout.addWidget(self.top_p_value, 4, 2)
        self.top_p_slider.valueChanged.connect(
            lambda v: self.top_p_value.setText(str(v/10))
        )
        
        # Temperature slider
        params_layout.addWidget(QLabel("Temperature:"), 5, 0)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setMinimum(1)
        self.temperature_slider.setMaximum(20)
        self.temperature_slider.setValue(10)
        params_layout.addWidget(self.temperature_slider, 5, 1)
        self.temperature_value = QLabel("1.0")
        params_layout.addWidget(self.temperature_value, 5, 2)
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_value.setText(str(v/10))
        )
        
        # Repetition Penalty slider
        params_layout.addWidget(QLabel("Repetition Penalty:"), 6, 0)
        self.repetition_penalty_slider = QSlider(Qt.Orientation.Horizontal)
        self.repetition_penalty_slider.setMinimum(10)
        self.repetition_penalty_slider.setMaximum(30)
        self.repetition_penalty_slider.setValue(10)
        params_layout.addWidget(self.repetition_penalty_slider, 6, 1)
        self.repetition_penalty_value = QLabel("1.0")
        params_layout.addWidget(self.repetition_penalty_value, 6, 2)
        self.repetition_penalty_slider.valueChanged.connect(
            lambda v: self.repetition_penalty_value.setText(str(v/10))
        )
        
        # Checkboxes
        checkbox_layout = QHBoxLayout()
        self.convert_style_checkbox = QCheckBox("Convert Style")
        self.anonymization_only_checkbox = QCheckBox("Anonymization Only")
        checkbox_layout.addWidget(self.convert_style_checkbox)
        checkbox_layout.addWidget(self.anonymization_only_checkbox)
        params_layout.addLayout(checkbox_layout, 7, 0, 1, 3)
        
        params_group.setLayout(params_layout)
        input_layout.addWidget(params_group)
        
        # Convert button
        self.convert_button = QPushButton("Convert Voice")
        self.convert_button.setMinimumHeight(40)
        self.convert_button.clicked.connect(self.start_conversion)
        input_layout.addWidget(self.convert_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        input_layout.addWidget(self.progress_bar)
        
        input_widget.setLayout(input_layout)
        splitter.addWidget(input_widget)
        
        # Output section
        output_widget = QWidget()
        output_layout = QVBoxLayout()
        
        output_title = QLabel("Output")
        output_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        output_layout.addWidget(output_title)
        
        # Output audio players
        output_players_layout = QHBoxLayout()
        
        # Stream Output widget
        self.stream_output = AudioPlayerWidget("Stream Output Audio")
        output_players_layout.addWidget(self.stream_output)
        
        # Full Output widget
        self.full_output = AudioPlayerWidget("Full Output Audio")
        output_players_layout.addWidget(self.full_output)
        
        output_layout.addLayout(output_players_layout)
        
        # Save output buttons
        save_buttons_layout = QHBoxLayout()
        
        self.save_stream_button = QPushButton("Save Stream Audio")
        self.save_stream_button.setEnabled(False)
        self.save_stream_button.clicked.connect(self.save_stream_output)
        
        self.save_full_button = QPushButton("Save Full Audio")
        self.save_full_button.setEnabled(False)
        self.save_full_button.clicked.connect(self.save_full_output)
        
        save_buttons_layout.addWidget(self.save_stream_button)
        save_buttons_layout.addWidget(self.save_full_button)
        
        output_layout.addLayout(save_buttons_layout)
        
        output_widget.setLayout(output_layout)
        splitter.addWidget(output_widget)
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready. Load models by converting.")
    
    def load_models(self, ar_checkpoint_path=None, cfm_checkpoint_path=None, compile=False):
        try:
            self.statusBar().showMessage("Loading models...")
            QApplication.processEvents()
            
            from hydra.utils import instantiate
            from omegaconf import DictConfig
            
            # Load configuration
            cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
            vc_wrapper = instantiate(cfg)
            
            # Load checkpoints
            vc_wrapper.load_checkpoints(
                ar_checkpoint_path=ar_checkpoint_path,
                cfm_checkpoint_path=cfm_checkpoint_path
            )
            
            # Transfer to device and set evaluation mode
            vc_wrapper.to(device)
            vc_wrapper.eval()
            
            # Setup AR caches
            vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
            
            # Compile model if requested
            if compile and hasattr(torch, "_inductor"):
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.triton.unique_kernel_names = True
                
                if hasattr(torch._inductor.config, "fx_graph_cache"):
                    torch._inductor.config.fx_graph_cache = True
                    
                vc_wrapper.compile_ar()
            
            self.statusBar().showMessage("Models loaded successfully")
            
            # Extend the original vc_wrapper.convert_voice_with_streaming method to support progress updates
            original_method = vc_wrapper.convert_voice_with_streaming
            
            def extended_convert_method(*args, progress_callback=None, stream_output_path=None, 
                                        full_output_path=None, **kwargs):
                # Call the original method with its normal arguments
                result = original_method(*args, **kwargs)
                
                # Save outputs to files if paths are provided
                if isinstance(result, tuple) and len(result) == 2:
                    stream_output, full_output = result
                    
                    if stream_output_path and hasattr(stream_output, "save"):
                        stream_output.save(stream_output_path)
                    
                    if full_output_path and hasattr(full_output, "save"):
                        full_output.save(full_output_path)
                
                return result
            
            # Replace the original method with our extended one
            vc_wrapper.convert_voice_with_streaming = extended_convert_method
            
            return vc_wrapper
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading models: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")
            return None
    
    def start_conversion(self):
        # Check if source and reference audio are selected
        source_path = self.source_audio.get_audio_path()
        reference_path = self.reference_audio.get_audio_path()
        
        if not source_path:
            QMessageBox.warning(self, "Warning", "Please select a source audio file")
            return
        
        if not reference_path:
            QMessageBox.warning(self, "Warning", "Please select a reference audio file")
            return
        
        # Load models if not already loaded
        if self.vc_wrapper is None:
            self.vc_wrapper = self.load_models()
            if self.vc_wrapper is None:
                return
        
        # Disable UI elements during conversion
        self.convert_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Converting...")
        
        # Prepare parameters
        params = {
            'source_path': source_path,
            'reference_path': reference_path,
            'diffusion_steps': self.diffusion_steps_slider.value(),
            'length_adjust': float(self.length_adjust_value.text()),
            'intelligibility_cfg': float(self.intelligibility_cfg_value.text()),
            'similarity_cfg': float(self.similarity_cfg_value.text()),
            'top_p': float(self.top_p_value.text()),
            'temperature': float(self.temperature_value.text()),
            'repetition_penalty': float(self.repetition_penalty_value.text()),
            'convert_style': self.convert_style_checkbox.isChecked(),
            'anonymization_only': self.anonymization_only_checkbox.isChecked()
        }
        
        # Create and start worker thread
        self.worker_thread = WorkerThread(self.vc_wrapper, params)
        self.worker_thread.progress_signal.connect(self.update_progress)
        self.worker_thread.finished_signal.connect(self.conversion_finished)
        self.worker_thread.error_signal.connect(self.conversion_error)
        self.worker_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def conversion_finished(self, stream_output_path, full_output_path):
        self.stream_output_path = stream_output_path
        self.full_output_path = full_output_path
        
        # Set audio files to players
        self.stream_output.set_audio_file(stream_output_path)
        self.full_output.set_audio_file(full_output_path)
        
        # Enable save buttons
        self.save_stream_button.setEnabled(True)
        self.save_full_button.setEnabled(True)
        
        # Reset UI
        self.convert_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Conversion completed successfully")
    
    def conversion_error(self, error_message):
        # Reset UI
        self.convert_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Error during conversion: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Error", f"Conversion failed: {error_message}")
    
    def save_stream_output(self):
        if self.stream_output_path:
            self.save_file(self.stream_output_path, "mp3")
    
    def save_full_output(self):
        if self.full_output_path:
            self.save_file(self.full_output_path, "wav")
    
    def save_file(self, source_path, file_type):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio File", "", f"Audio Files (*.{file_type})"
        )
        
        if file_path:
            # Ensure file has the correct extension
            if not file_path.endswith(f".{file_type}"):
                file_path += f".{file_type}"
            
            # Copy the file
            try:
                import shutil
                shutil.copy2(source_path, file_path)
                QMessageBox.information(self, "Success", f"File saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def closeEvent(self, event):
        # Clean up temporary files
        if self.stream_output_path and os.path.exists(self.stream_output_path):
            try:
                os.remove(self.stream_output_path)
            except:
                pass
        
        if self.full_output_path and os.path.exists(self.full_output_path):
            try:
                os.remove(self.full_output_path)
            except:
                pass
        
        event.accept()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = VoiceConverterApp()
    window.show()
    sys.exit(app.exec())
