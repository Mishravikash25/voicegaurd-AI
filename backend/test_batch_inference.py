import os
import sys
from pathlib import Path

# Add backend root to sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))

from ml_system.inference.predict import VoiceAuthenticator

def test_inference_batch():
    authenticator = VoiceAuthenticator()
    
    real_dir = backend_dir / "dataset" / "KAGGLE" / "AUDIO" / "REAL"
    fake_dir = backend_dir / "dataset" / "KAGGLE" / "AUDIO" / "FAKE"

    real_samples = list(real_dir.glob("*.wav"))[:5]
    fake_samples = list(fake_dir.glob("*.wav"))[:5]

    print("=" * 65)
    print("      VOICEGUARD AI - BATCH FORENSIC MODEL TESTING")
    print("=" * 65)

    print("\n--- Testing REAL Audio Samples ---")
    for sample in real_samples:
        try:
            res = authenticator.analyze(str(sample))
            print(f"File: {sample.name:<25} | Verdict: {res['verdict']:<8} | Fraud Prob: {res['fraud_probability']:>5.1f}% | Similarity: {res['similarity_score']:>5.1f}%")
        except Exception as e:
            print(f"File: {sample.name:<25} | Error: {e}")

    print("\n--- Testing FAKE / DEEPFAKE Audio Samples ---")
    for sample in fake_samples:
        try:
            res = authenticator.analyze(str(sample))
            print(f"File: {sample.name:<25} | Verdict: {res['verdict']:<8} | Fraud Prob: {res['fraud_probability']:>5.1f}% | Similarity: {res['similarity_score']:>5.1f}%")
        except Exception as e:
            print(f"File: {sample.name:<25} | Error: {e}")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    test_inference_batch()
