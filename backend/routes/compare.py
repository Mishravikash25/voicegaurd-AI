from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import shutil
import tempfile
import sys
import numpy as np
from numpy.linalg import norm
from typing import Dict, Any, Optional

# Ensure sys.path includes backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_system.inference.predict import authenticator
from ml_system.features.preprocess import preprocessor
from ml_system.features.extract import extractor

router = APIRouter()

def extract_speaker_embedding(audio_path: str) -> Optional[np.ndarray]:
    """
    Extracts a feature-wise standardized speaker timbre embedding.
    Scales frequency features and zeroes out volume gain offsets.
    """
    cleaned = preprocessor.process(audio_path)
    if cleaned is None or len(cleaned) == 0:
        return None

    extractor.scaler = None
    vec = extractor.extract_features(cleaned)
    if vec is None:
        return None

    vec_norm = vec.copy()
    # 1. Zero out gain/volume offsets (MFCC 0 means & stds, RMS energy)
    vec_norm[0] = 0.0
    vec_norm[20] = 0.0
    vec_norm[40] = 0.0
    vec_norm[60] = 0.0
    vec_norm[80] = 0.0
    vec_norm[100] = 0.0
    vec_norm[126] = 0.0
    vec_norm[127] = 0.0

    # 2. Scale spectral features (Centroid & Bandwidth in kHz) so they don't overpower MFCC timbre
    vec_norm[120:124] /= 1000.0

    # 3. Z-score standardize across vector dimensions
    mean_val = np.mean(vec_norm)
    std_val = np.std(vec_norm)
    z_vec = (vec_norm - mean_val) / (std_val + 1e-8)

    return z_vec

@router.post("/compare")
async def compare_two_audios(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    1-to-1 Audio Comparison Endpoint:
    Compares two uploaded audio files for pairwise acoustic similarity, speaker verification, and deepfake detection.
    """
    if not file_a.filename or not file_b.filename:
        raise HTTPException(status_code=400, detail="Both audio files must be provided.")

    # Create temporary files
    suffix_a = os.path.splitext(file_a.filename)[1] or ".wav"
    suffix_b = os.path.splitext(file_b.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a) as tmp_a:
        shutil.copyfileobj(file_a.file, tmp_a)
        temp_a_path = tmp_a.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b) as tmp_b:
        shutil.copyfileobj(file_b.file, tmp_b)
        temp_b_path = tmp_b.name

    try:
        # 1. Forensic Deepfake Analysis for both files
        res_a = authenticator.analyze(temp_a_path)
        res_b = authenticator.analyze(temp_b_path)

        # 2. Extract standardized speaker embeddings
        v_a = extract_speaker_embedding(temp_a_path)
        v_b = extract_speaker_embedding(temp_b_path)

        if v_a is not None and v_b is not None:
            euc_dist = float(norm(v_a - v_b))
            # RBF exponential decay formula with sigma=1.35
            rbf_score = np.exp(-(euc_dist ** 2) / (2 * (1.35 ** 2))) * 100.0
            pair_similarity = round(float(max(0.0, min(100.0, rbf_score))), 2)
        else:
            pair_similarity = 0.0

        # Determine speaker match verdict
        if res_a["verdict"] == "FRAUD" or res_b["verdict"] == "FRAUD":
            speaker_verdict = "SYNTHETIC_DEEPFAKE_DETECTED"
        elif pair_similarity >= 68.0:
            speaker_verdict = "MATCH_SAME_SPEAKER"
        elif pair_similarity >= 45.0:
            speaker_verdict = "INCONCLUSIVE_POSSIBLE_MATCH"
        else:
            speaker_verdict = "DIFFERENT_SPEAKERS"

        return {
            "file_a": {
                "filename": file_a.filename,
                "verdict": res_a["verdict"],
                "fraud_probability": res_a["fraud_probability"],
                "similarity_score": res_a["similarity_score"]
            },
            "file_b": {
                "filename": file_b.filename,
                "verdict": res_b["verdict"],
                "fraud_probability": res_b["fraud_probability"],
                "similarity_score": res_b["similarity_score"]
            },
            "pair_acoustic_similarity": pair_similarity,
            "speaker_verdict": speaker_verdict
        }

    except Exception as e:
        print(f"1-to-1 Comparison Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    finally:
        if os.path.exists(temp_a_path):
            os.remove(temp_a_path)
        if os.path.exists(temp_b_path):
            os.remove(temp_b_path)
