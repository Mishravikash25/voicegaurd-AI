import sys
import os
import numpy as np
import librosa
import soundfile as sf

# Add backend to path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.features import feature_extractor

def generate_test_audio(filename="test_features_audio.wav", duration=1.0, sr=16000):
    """Generates a 1kHz sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    # Mix of 440Hz and 1000Hz sine waves
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)
    sf.write(filename, audio, sr)
    print(f"Generated {filename}")
    return filename

def run_test():
    test_file = generate_test_audio()
    
    try:
        # Load audio using librosa to match extraction sr
        audio, _ = librosa.load(test_file, sr=16000)
        
        print("Starting Neural Feature Extraction Pipeline...")
        # Note: Features are extracted per STFT frame
        feature_matrix = feature_extractor.extract_all_features(audio)
        
        print(f"Success! Feature Matrix Shape: {feature_matrix.shape}")
        print(f"Number of frames: {feature_matrix.shape[0]}")
        print(f"Feature Dimensions: {feature_matrix.shape[1]}")
        
        # Verify Dimension count (13*3 + 4 + 1 = 44)
        if feature_matrix.shape[1] == 44:
            print("Dimension Verification (44D): PASSED")
        else:
            print(f"Dimension Verification (44D): FAILED (Got {feature_matrix.shape[1]})")

        # Basic check for non-zero features
        if np.any(feature_matrix):
            print("Non-zero Signal Detection: PASSED")
        else:
            print("Non-zero Signal Detection: FAILED")

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

if __name__ == "__main__":
    run_test()
