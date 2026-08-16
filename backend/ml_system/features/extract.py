import logging
import librosa
import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts acoustic features from cleaned audio signals.
    Outputs a fixed-length, 1D normalized feature vector suitable for ML.
    """
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 20, hop_length: int = 512, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # A scaler to normalize the final 1D feature vector
        # Note: In production training, you fit the scaler on the whole training set beforehand.
        # Here we provide standard scaling on the extracted vector to ensure numerical stability.
        self.scaler = StandardScaler()

    def calculate_spectral_entropy(self, spectrogram: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Approximates the Shannon entropy of the spectrogram per frame.
        """
        # Normalize the spectrogram so that sum over frequency bins is 1 for each frame
        psd_norm = spectrogram / (np.sum(spectrogram, axis=0, keepdims=True) + epsilon)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(psd_norm * np.log2(psd_norm + epsilon), axis=0)
        return entropy

    def extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates:
        1. MFCC (13-20 typically, using 20 here)
        2. Delta MFCC
        3. Delta-Delta MFCC
        4. Spectral Centroid
        5. Spectral Bandwidth
        6. Zero Crossing Rate (ZCR)
        7. RMS Energy
        8. Spectral Entropy
        """
        try:
            # Short-Time Fourier Transform used by multiple features
            S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
            
            # 1. MFCC (returns frame-wise 2D array: [n_mfcc, time_steps])
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc, hop_length=self.hop_length, n_fft=self.n_fft)
            
            # 2 & 3. Delta and Delta-Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs, order=1)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # 4. Spectral Centroid
            centroid = librosa.feature.spectral_centroid(S=S, sr=self.sample_rate)[0]
            
            # 5. Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=self.sample_rate)[0]
            
            # 6. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)[0]
            
            # 7. RMS Energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            
            # 8. Spectral Entropy
            entropy = self.calculate_spectral_entropy(S)

            # --- Aggregation ---
            # To feed ML models (like SVMs, Random Forests, or Dense NNs), 
            # we need a FIXED structure (1D vector per audio file).
            # We achieve this by calculating global statistics (mean, std) across the time axis.
            
            features = []
            
            # Helper function to append mean and std
            def add_stats(feat_array: np.ndarray):
                if feat_array.ndim == 1:
                    features.extend([np.mean(feat_array), np.std(feat_array)])
                else:
                    features.extend(np.mean(feat_array, axis=1).tolist())
                    features.extend(np.std(feat_array, axis=1).tolist())

            add_stats(mfccs)         # 20 means + 20 stds = 40
            add_stats(delta_mfccs)   # 40
            add_stats(delta2_mfccs)  # 40
            add_stats(centroid)      # 2
            add_stats(bandwidth)     # 2
            add_stats(zcr)           # 2
            add_stats(rms)           # 2
            add_stats(entropy)       # 2
            
            # Total features theoretically: (20*2)*3 + 2*5 = 120 + 10 = 130 dimensions
            feature_vector = np.array(features, dtype=float)
            
            # Ensure no NaNs or Infs from calculation anomalies (e.g. total silence passing through)
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                logger.warning("NaNs or Infs detected in feature vector. Replacing with 0.")
                feature_vector = np.nan_to_num(feature_vector)

            # --- Normalization ---
            # Reshape for sklearn scaler (expects 2D array [samples, features])
            # Since we are normalizing a SINGLE vector independently here, it ensures feature bounds
            # For a production pipeline over a whole dataset, you'd stack all vectors and fit the scaler once.
            if self.scaler is not None:
                vector_2d = feature_vector.reshape(-1, 1)
                normalized_vector = self.scaler.fit_transform(vector_2d).flatten()
            else:
                normalized_vector = feature_vector

            return normalized_vector

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def extract_mel_spectrogram(self, signal):
        """
        Extracts a 2D Log-Mel Spectrogram for use in Convolutional Neural Networks (CNNs).
        Shape: (Mels, TimeFrames)
        """
        try:
            melspec = librosa.feature.melspectrogram(
                y=signal, 
                sr=self.sample_rate, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                n_mels=128 # Standard for Deep Learning
            )
            # Convert to log scale (dB)
            log_melspec = librosa.power_to_db(melspec, ref=np.max)

            # Pad or truncate to fixed 150 time frames
            target_frames = 150
            if log_melspec.shape[1] < target_frames:
                pad_width = target_frames - log_melspec.shape[1]
                log_melspec = np.pad(log_melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                log_melspec = log_melspec[:, :target_frames]

            return log_melspec
        except Exception as e:
            logger.error(f"Error extracting Mel-Spectrogram: {e}")
            return None

    def extract_sequence_features(self, signal):
        """
        Extracts a sequence of MFCC vectors for use in Recurrent Neural Networks (LSTMs).
        Instead of averaging them flat, it returns the 2D array: (TimeFrames, Features)
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=signal, 
                sr=self.sample_rate, 
                n_mfcc=40, # Usually 40 for LSTMs
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Standardize sequence length by padding or truncating 
            # (assuming standard 3 second audio = ~94 frames)
            target_frames = 100
            
            if mfccs.shape[1] < target_frames:
                pad_width = target_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :target_frames]
                
            # PyTorch expects (TimeFrames, Features), so transpose:
            return mfccs.T
        except Exception as e:
            logger.error(f"Error extracting Sequence Features: {e}")
            return None

# Reusable singleton instance
extractor = FeatureExtractor()

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger.info("Initializing mock 16kHz audio sample...")
    sr = 16000
    t = np.linspace(0, 2, sr * 2) # 2 seconds
    mock_audio = np.sin(2 * np.pi * 440 * t) # 440Hz tone
    
    logger.info("Running Feature Extraction...")
    feat_extractor = FeatureExtractor(sample_rate=sr)
    
    feature_vec = feat_extractor.extract_features(mock_audio)
    
    if feature_vec is not None:
        logger.info(f"Successfully extracted fixed-length vector of shape: {feature_vec.shape}")
        logger.info(f"Vector Mean: {np.mean(feature_vec):.4f}, Std: {np.std(feature_vec):.4f}")
    else:
        logger.error("Failed to extract features from mock audio.")
