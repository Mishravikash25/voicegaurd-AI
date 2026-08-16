import librosa
import numpy as np
from scipy.signal import butter, lfilter
import os
from typing import Tuple, List

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.win_length = int(0.025 * sample_rate)  # 25ms
        self.hop_length = int(0.010 * sample_rate)  # 10ms

    def load_audio(self, file_path: str) -> np.ndarray:
        """Loads and resamples audio to defined sample rate."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio

    def apply_bandpass_filter(self, data: np.ndarray, lowcut: float = 300.0, highcut: float = 3400.0, order: int = 5) -> np.ndarray:
        """Applies a Butterworth bandpass filter."""
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def normalize_amplitude(self, data: np.ndarray) -> np.ndarray:
        """Normalizes audio amplitude to [-1, 1]."""
        if np.max(np.abs(data)) > 0:
            return data / np.max(np.abs(data))
        return data

    def segment_frames(self, data: np.ndarray) -> np.ndarray:
        """Segments audio into overlapping frames and applies Hamming window."""
        frames = librosa.util.frame(data, frame_length=self.win_length, hop_length=self.hop_length).T
        hamming_window = np.hamming(self.win_length)
        return frames * hamming_window

    def apply_cmn(self, frames: np.ndarray) -> np.ndarray:
        """Applies Cepstral Mean Normalization (CMN)."""
        # For raw frames, we normalize across the time axis for each frame
        # In a real MFCC pipeline, this is done on cepstral coefficients, 
        # but here we apply it to the framed signal as requested.
        mean = np.mean(frames, axis=0)
        return frames - mean

    def process_signal(self, audio_signal: np.ndarray) -> np.ndarray:
        """Executes the forensic pipeline on an in-memory signal."""
        # 1. Bandpass Filter
        filtered_audio = self.apply_bandpass_filter(audio_signal)
        
        # 2. Normalize
        normalized_audio = self.normalize_amplitude(filtered_audio)
        
        # 3. Framing & Windowing
        frames = self.segment_frames(normalized_audio)
        
        # 4. CMN
        processed_frames = self.apply_cmn(frames)
        
        return processed_frames

    def process_pipeline(self, file_path: str) -> np.ndarray:
        """Executes the full preprocessing forensic pipeline."""
        # 1. Load & Resample
        audio = self.load_audio(file_path)
        return self.process_signal(audio)

# Reusable instance
preprocessor = AudioPreprocessor()
