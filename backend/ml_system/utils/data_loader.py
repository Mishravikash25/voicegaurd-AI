import os
import logging
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings from librosa (e.g., about PySoundFile falling back to audioread for mp3/mp4)
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

class AudioDataLoader:
    """
    A robust data loader for the VoiceGuard AI machine learning system.
    Handles loading, validating, and labeling of audio files (.wav, .mp3, .mp4).
    """

    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.mp4'}

    def __init__(self, sample_rate: int = 16000, duration: Optional[float] = None):
        """
        Args:
            sample_rate (int): Target sampling rate to resample all audio to.
            duration (float, optional): Maximum duration of audio to load in seconds.
                                        If None, loads the entire audio file.
        """
        self.sample_rate = sample_rate
        self.duration = duration

    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Safely attempts to load an audio file.

        Args:
            file_path (str): The absolute or relative path to the audio file.

        Returns:
            Optional[np.ndarray]: The numpy array of the audio signal, or None if corrupted/failed.
        """
        try:
            # librosa.load naturally handles resampling and mono conversion
            signal, _ = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                duration=self.duration, 
                mono=True
            )
            return signal
        except Exception as e:
            logger.error(f"Failed to load file '{file_path}': {e}")
            return None

    def load_dataset(self, base_dir: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Traverses a directory structure expecting 'genuine' and 'fake' subfolders.
        Extracts raw audio arrays and numeric labels (0 for genuine, 1 for fake).
        Skips invalid or corrupted files smoothly.

        Args:
            base_dir (str): Path to the dataset folder (e.g., 'data/raw/').

        Returns:
            Tuple[List[np.ndarray], List[int]]: X (audio signals) and y (labels).
        """
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.error(f"Directory not found: {base_path}")
            return [], []

        X = []
        y = []

        # Define class mapping
        class_mapping = {
            'genuine': 0,
            'real': 0,
            'REAL': 0,
            'fake': 1,
            'FAKE': 1
        }

        total_files = 0
        successful_loads = 0

        for class_name, label in class_mapping.items():
            class_dir = base_path / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class folder '{class_name}' missing in {base_path}. Skipping.")
                continue

            logger.info(f"Scanning '{class_name}' folder...")
            
            # Scans recursively for supported extensions
            for ext in self.SUPPORTED_EXTENSIONS:
                for file_path in class_dir.rglob(f"*{ext}"):
                    total_files += 1
                    
                    logger.debug(f"Loading {file_path.name}...")
                    signal = self.load_audio_file(str(file_path))
                    
                    if signal is not None:
                        # Only add successfully loaded, non-empty arrays
                        if len(signal) > 0:
                            X.append(signal)
                            y.append(label)
                            successful_loads += 1
                        else:
                            logger.warning(f"File '{file_path.name}' loaded as empty array. Skipping.")

        logger.info(f"Data loading complete. Successfully loaded {successful_loads}/{total_files} files.")
        
        return X, y

# Example usage/test execution block
if __name__ == "__main__":
    import sys
    # When run directly, try to hit the data/raw folder
    project_root = Path(__file__).resolve().parent.parent
    raw_data_path = project_root / 'data' / 'raw'
    
    logger.info(f"Testing the AudioDataLoader targeting '{raw_data_path}'...")
    
    loader = AudioDataLoader(sample_rate=16000, duration=5.0)
    X_train, y_train = loader.load_dataset(str(raw_data_path))
    
    logger.info(f"Loaded {len(X_train)} samples total.")
