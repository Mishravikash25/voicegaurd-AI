import sys
import os
import numpy as np
import librosa
import soundfile as sf

# Add backend to path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.preprocess import preprocessor

def generate_test_audio(filename="test_audio.wav", duration=1.0, sr=16000):
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
        print("Starting Preprocessing Pipeline...")
        frames = preprocessor.process_pipeline(test_file)
        
        print(f"Success! Processed Frames Shape: {frames.shape}")
        print(f"Number of frames: {frames.shape[0]}")
        print(f"Frame length: {frames.shape[1]} samples")
        
        # Verify CMN (mean should be close to 0 per band)
        frame_mean = np.mean(frames, axis=0)
        print(f"Average frame mean after CMN: {np.mean(np.abs(frame_mean)):.6f}")
        
        if np.mean(np.abs(frame_mean)) < 1e-5:
            print("CMN Verification: PASSED")
        else:
            print("CMN Verification: FAILED (Mean not zero)")

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

if __name__ == "__main__":
    run_test()
