import os
import sys
import numpy as np
from pathlib import Path
from numpy.linalg import norm

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

from ml_system.inference.predict import VoiceAuthenticator
from ml_system.features.preprocess import preprocessor
from ml_system.features.extract import extractor

def compare_two_audio_files(file1_path: str, file2_path: str):
    """
    Compares two audio files:
    1. Computes acoustic feature vectors and Cosine Similarity between File 1 and File 2.
    2. Runs deepfake forensic model analysis on both files to output Fraud Probability and Verdict.
    """
    if not os.path.exists(file1_path):
        raise FileNotFoundError(f"File 1 not found: {file1_path}")
    if not os.path.exists(file2_path):
        raise FileNotFoundError(f"File 2 not found: {file2_path}")

    authenticator = VoiceAuthenticator()

    # 1. Analyze File 1
    res1 = authenticator.analyze(file1_path)
    
    # 2. Analyze File 2
    res2 = authenticator.analyze(file2_path)

    # 3. Compute 1-to-1 Cosine Similarity between acoustic feature vectors
    audio1 = preprocessor.process(file1_path)
    audio2 = preprocessor.process(file2_path)

    extractor.scaler = None
    vec1 = extractor.extract_features(audio1)
    vec2 = extractor.extract_features(audio2)

    if vec1 is not None and vec2 is not None:
        # Cosine similarity formula
        dot_prod = np.dot(vec1, vec2)
        norm_val = norm(vec1) * norm(vec2)
        cosine_sim = (dot_prod / norm_val) if norm_val != 0 else 0.0
        similarity_percentage = round(float(max(0.0, min(1.0, cosine_sim)) * 100), 2)
    else:
        similarity_percentage = 0.0

    report = {
        "file1": {
            "path": file1_path,
            "filename": os.path.basename(file1_path),
            "verdict": res1["verdict"],
            "fraud_probability": res1["fraud_probability"],
            "model_similarity_score": res1["similarity_score"]
        },
        "file2": {
            "path": file2_path,
            "filename": os.path.basename(file2_path),
            "verdict": res2["verdict"],
            "fraud_probability": res2["fraud_probability"],
            "model_similarity_score": res2["similarity_score"]
        },
        "pair_acoustic_similarity": similarity_percentage
    }

    return report

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        f1, f2 = sys.argv[1], sys.argv[2]
        res = compare_two_audio_files(f1, f2)
        print("\n" + "=" * 60)
        print("     VOICEGUARD AI - 1-TO-1 AUDIO COMPARISON REPORT")
        print("=" * 60)
        print(f"File 1: {res['file1']['filename']}")
        print(f"  |- Verdict:           {res['file1']['verdict']}")
        print(f"  |- Fraud Probability: {res['file1']['fraud_probability']}%")
        print(f"\nFile 2: {res['file2']['filename']}")
        print(f"  |- Verdict:           {res['file2']['verdict']}")
        print(f"  |- Fraud Probability: {res['file2']['fraud_probability']}%")
        print(f"\nPair Acoustic Similarity: {res['pair_acoustic_similarity']}%")
        print("=" * 60 + "\n")
    else:
        print("Usage: python compare_audio_files.py <path_to_audio1> <path_to_audio2>")
