import librosa
import numpy as np
from typing import Tuple

class FeatureExtractor:
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_mfcc_suite(self, audio: np.ndarray) -> np.ndarray:
        """Extracts MFCCs, Delta, and Delta-Delta features."""
        # 1. Base MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        
        # 2. Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # 3. Delta-Delta MFCCs
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine (39, Time)
        return np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extracts Spectral Centroid, Bandwidth, and Entropy."""
        # 1. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        
        # 2. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        
        # 3. Spectral Flatness (Related to Entropy)
        flatness = librosa.feature.spectral_flatness(y=audio)
        
        # 4. RMS Energy
        rmse = librosa.feature.rms(y=audio)
        
        return np.vstack([centroid, bandwidth, flatness, rmse])

    def calculate_spectral_entropy(self, audio: np.ndarray) -> np.ndarray:
        """Calculates normalized spectral entropy per frame."""
        # Compute Power Spectrogram
        stft = np.abs(librosa.stft(audio))
        psd = stft**2
        
        # Normalize PSD to get a probability distribution
        psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
        
        # Calculate Entropy: -sum(p * log2(p))
        # Add epsilon to avoid log(0)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=0) / np.log2(psd.shape[0])
        
        # Reshape to (1, Time) to match other features
        return entropy.reshape(1, -1)

    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """Extracts and combines all features into a single matrix (Time, Dimensions)."""
        # Optimization: Compute STFT/Spectrogram once and reuse
        stft = np.abs(librosa.stft(audio))
        psd = stft**2
        
        # 1. MFCC Suite
        # librosa.feature.mfcc can take S (spectrogram) directly
        mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft), sr=self.sample_rate, n_mfcc=self.n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfcc_suite = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

        # 2. Spectral Suite (Reuse S or psd where possible)
        centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sample_rate)
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=self.sample_rate)
        flatness = librosa.feature.spectral_flatness(S=stft)
        rmse = librosa.feature.rms(S=stft)
        spectral_suite = np.vstack([centroid, bandwidth, flatness, rmse])
        
        # 3. Entropy (Reuse psd)
        psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=0) / np.log2(psd.shape[0])
        entropy = entropy.reshape(1, -1)
        
        # Combine all: (Dimensions, Time)
        combined = np.vstack([mfcc_suite, spectral_suite, entropy])
        
        # Transpose to (Time, Dimensions)
        return combined.T

# Reusable instance
feature_extractor = FeatureExtractor()
