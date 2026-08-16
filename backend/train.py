import os
import sys
import numpy as np
from typing import List

# Ensure we can import from services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.preprocess import preprocessor
from services.features import feature_extractor
from services.model import forensic_model

def train_forensic_pipeline(dataset_path: str):
    """
    Automated training pipeline for VoiceGuard AI.
    Processes a directory of audio files to build a GMM-based forensic model.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    audio_files = [
        f for f in os.listdir(dataset_path) 
        if f.endswith(('.wav', '.mp3'))
    ]

    if not audio_files:
        print(f"Error: No .wav or .mp3 files found in '{dataset_path}'.")
        return

    print(f"Found {len(audio_files)} audio files. Starting forensic extraction...")

    all_features: List[np.ndarray] = []

    for i, filename in enumerate(audio_files):
        file_path = os.path.join(dataset_path, filename)
        print(f"[{i+1}/{len(audio_files)}] Processing: {filename}...", end="\r")
        
        try:
            # 1. Load & Preprocess
            audio = preprocessor.load_audio(file_path)
            filtered = preprocessor.apply_bandpass_filter(audio)
            normalized = preprocessor.normalize_amplitude(filtered)
            
            # 2. Extract Features (44D)
            features = feature_extractor.extract_all_features(normalized)
            
            all_features.append(features)
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")

    print(f"\nExtraction complete. Building forensic feature matrix...")

    if not all_features:
        print("Error: No features extracted. Training aborted.")
        return

    # Concatenate all features: (Total Frames, 44)
    final_feature_matrix = np.vstack(all_features)
    print(f"Final Feature Matrix Shape: {final_feature_matrix.shape}")

    # 3. Train GMM Model
    try:
        forensic_model.train(final_feature_matrix)
        
        # 4. Save Model
        forensic_model.save_model()
        print("\nForensic Training Successful!")
        print(f"Model saved to: {forensic_model.model_path}")
        
    except Exception as e:
        print(f"\nTraining Error: {str(e)}")

if __name__ == "__main__":
    # Default behavior: look for 'data' folder or accept argument
    path = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    print("=== VoiceGuard AI Forensic Training Service ===")
    train_forensic_pipeline(path)
