"""
Configuration parameters for VoiceGuard AI Machine Learning Pipeline.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Audio Processing Configuration
AUDIO_PARAMS = {
    "sample_rate": 22050,
    "duration": None, # or max length in seconds, e.g. 5.0
    "n_mfcc": 44,
    "n_fft": 2048,
    "hop_length": 512
}

# Training Configuration (Scikit-Learn GMM initially)
TRAINING_PARAMS = {
    "gmm_components": 16,
    "covariance_type": "diag",
    "max_iter": 200,
    "random_state": 42
}

# Future: PyTorch related configs
PYTORCH_CONFIG = {
    "use_pytorch": False,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 50
}
