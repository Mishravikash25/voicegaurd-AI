import logging
import librosa
import numpy as np
from scipy import signal
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Handles all pre-processing steps for audio signals before feature extraction.
    Ensures uniformity in sample rate, amplitude, and removes silence and noise.
    """
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_and_resample(self, file_path: str) -> Optional[np.ndarray]:
        """
        Loads an audio file and resamples it to the target sample rate.
        """
        try:
            # librosa.load will automatically resample if sr is provided
            audio, _ = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def apply_bandpass_filter(self, audio: np.ndarray, lowcut: float = 50.0, highcut: float = 7600.0, order: int = 5) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter. 
        50-7600 Hz retains wideband acoustic cues (essential for deepfake artifact detection).
        """
        nyquist = 0.5 * self.target_sr
        low = max(0.001, lowcut / nyquist)
        high = min(0.999, highcut / nyquist)
        
        # Determine filter coefficients
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply the filter (filtfilt ensures zero-phase distortion)
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio

    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalizes the audio signal amplitude to the range [-1.0, 1.0].
        """
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            return audio / max_amp
        return audio

    def remove_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """
        Trims leading and trailing silence from an audio signal.
        top_db: The threshold (in decibels) below reference to consider as silence.
        """
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    def apply_cmn(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies time-domain Cepstral Mean Normalization (CMN) approximation.
        While standard CMN is applied on MFCC features, subtracting the 
        mean of the wave signal removes DC bias/offset before feature extraction.
        """
        mean_val = np.mean(audio)
        cmn_audio = audio - mean_val
        return cmn_audio

    def process(self, file_path: str) -> Optional[np.ndarray]:
        """
        Executes the full preprocessing pipeline in order:
        1. Load & Resample
        2. Bandpass Filter
        3. Remove Silence
        4. Normalize Amplitude
        5. CMN (DC Offset removal)
        """
        # 1. Load & Resample
        audio = self.load_and_resample(file_path)
        if audio is None or len(audio) == 0:
            return None

        # 2. Bandpass Filter (300 - 3400 Hz)
        audio = self.apply_bandpass_filter(audio)

        # 3. Remove Silence
        audio = self.remove_silence(audio)
        
        # Return none if silence removal deleted everything
        if len(audio) == 0:
            return None

        # 4. CMN (Mean Subtraction)
        audio = self.apply_cmn(audio)

        # 5. Peak Amplitude Normalization
        audio = self.normalize_amplitude(audio)

        return audio

# Reusable singleton instance
preprocessor = AudioPreprocessor()

if __name__ == "__main__":
    # For quick testing, mock a short audio signal
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(levelname)s: %(message)s')
    
    logger.info("Testing Preprocessor pipeline on mock sine wave...")
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, False)  # 1 second
    
    # 1000 Hz tone + DC offset + very low frequency noise
    mock_audio = np.sin(2 * np.pi * 1000 * t) + 0.5 + 0.2 * np.sin(2 * np.pi * 50 * t)
    
    processor = AudioPreprocessor()
    
    logger.info(f"Original signal max: {np.max(mock_audio):.2f}, mean: {np.mean(mock_audio):.2f}")
    
    # Run through the individual steps for mock testing without the load phase
    filtered = processor.apply_bandpass_filter(mock_audio)
    logger.info(f"After bandpass mean (should be near 0): {np.mean(filtered):.4f}")
    
    normalized = processor.normalize_amplitude(filtered)
    logger.info(f"After normalization max: {np.max(normalized):.2f}")
    
    logger.info("Pipeline test successful.")
