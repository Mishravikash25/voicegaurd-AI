import os
from pathlib import Path

# Base ML System path
BASE_DIR = Path(__file__).resolve().parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Paths
MODELS_DIR = BASE_DIR / "models"

# Audio Settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MFCC = 40

# Feature Extraction Settings
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Training Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
