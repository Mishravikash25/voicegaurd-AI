from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import os
import shutil
import tempfile
import sys
import json
import zipfile
import numpy as np
from numpy.linalg import norm
from typing import Dict, Any, List, Optional
from pathlib import Path

# Ensure sys.path includes backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_system.inference.predict import authenticator
from ml_system.features.preprocess import preprocessor
from ml_system.features.extract import extractor

router = APIRouter()

PROFILES_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "enrolled_profiles"
KAGGLE_REAL_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "dataset" / "KAGGLE" / "AUDIO" / "REAL"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_REAL_DIR.mkdir(parents=True, exist_ok=True)

# Default pre-enrolled dataset profiles from local Kaggle dataset
DEFAULT_PROFILES = {
    "Joe Biden": "biden-original.wav",
    "Barack Obama": "obama-original.wav",
    "Donald Trump": "trump-original.wav",
    "Elon Musk": "musk-original.wav",
    "Linus Torvalds": "linus-original.wav",
    "Margot Robbie": "margot-original.wav"
}

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

def get_profile_vector(speaker_name: str) -> Optional[np.ndarray]:
    """Helper to extract or load feature vector for a speaker."""
    safe_name = "".join([c for c in speaker_name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    npy_path = PROFILES_DIR / f"{safe_name}.npy"

    if npy_path.exists():
        try:
            return np.load(npy_path)
        except Exception:
            pass

    # Check default Kaggle profiles
    if speaker_name in DEFAULT_PROFILES:
        audio_filename = DEFAULT_PROFILES[speaker_name]
        audio_path = KAGGLE_REAL_DIR / audio_filename
        if audio_path.exists():
            return extract_speaker_embedding(str(audio_path))

    return None

@router.get("/profiles")
async def list_voice_profiles() -> Dict[str, Any]:
    """Lists all available enrolled voice profiles (default dataset + user custom profiles)."""
    profiles = list(DEFAULT_PROFILES.keys())

    # Add custom enrolled profiles from PROFILES_DIR
    if PROFILES_DIR.exists():
        for file in PROFILES_DIR.glob("*.json"):
            try:
                with open(file, "r") as f:
                    meta = json.load(f)
                    if meta.get("speaker_name") and meta["speaker_name"] not in profiles:
                        profiles.append(meta["speaker_name"])
            except Exception:
                pass

    return {
        "status": "success",
        "total_profiles": len(profiles),
        "profiles": profiles
    }

@router.post("/profiles/enroll")
async def enroll_voice_profile(
    speaker_name: str = Form(...),
    description: Optional[str] = Form("Custom User Enrolled Profile"),
    files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Enrolls a new target voice profile supporting MULTIPLE audio files at once!
    Extracts acoustic feature embeddings across all uploaded samples, averages them,
    and saves the composite speaker profile into enrolled_profiles and dataset directory.
    """
    if not speaker_name or not speaker_name.strip():
        raise HTTPException(status_code=400, detail="Speaker name is required.")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one audio file is required for enrollment.")

    safe_name = "".join([c for c in speaker_name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    vectors = []
    processed_filenames = []

    temp_dir = tempfile.mkdtemp()

    try:
        for idx, file_item in enumerate(files):
            if not file_item.filename:
                continue

            suffix = os.path.splitext(file_item.filename)[1].lower() or ".wav"
            temp_path = os.path.join(temp_dir, f"sample_{idx}{suffix}")

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file_item.file, buffer)

            # If zip archive uploaded in profile files
            if suffix == ".zip":
                with zipfile.ZipFile(temp_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
                    for extracted_file in Path(temp_dir).rglob("*"):
                        if extracted_file.is_file() and extracted_file.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
                            vec = extract_speaker_embedding(str(extracted_file))
                            if vec is not None:
                                vectors.append(vec)
                                processed_filenames.append(extracted_file.name)
                                # Copy to Kaggle REAL dataset
                                target_audio_path = KAGGLE_REAL_DIR / f"user_{safe_name}_{extracted_file.name}"
                                shutil.copy(str(extracted_file), target_audio_path)
            else:
                vec = extract_speaker_embedding(temp_path)
                if vec is not None:
                    vectors.append(vec)
                    processed_filenames.append(file_item.filename)
                    # Copy to Kaggle REAL dataset for retrain
                    target_audio_path = KAGGLE_REAL_DIR / f"user_{safe_name}_{idx}{suffix}"
                    shutil.copy(temp_path, target_audio_path)

        if len(vectors) == 0:
            raise HTTPException(status_code=500, detail="Failed to extract feature vectors from provided audio files.")

        # Compute average speaker embedding vector across all samples!
        composite_vector = np.mean(vectors, axis=0)

        # Save feature vector and metadata
        npy_path = PROFILES_DIR / f"{safe_name}.npy"
        meta_path = PROFILES_DIR / f"{safe_name}.json"

        np.save(npy_path, composite_vector)
        
        meta = {
            "speaker_name": speaker_name.strip(),
            "description": description,
            "sample_count": len(vectors),
            "filenames": processed_filenames,
            "vector_shape": list(composite_vector.shape)
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "status": "success",
            "message": f"Successfully enrolled voice profile for '{speaker_name}' with {len(vectors)} audio sample(s)!",
            "speaker_name": speaker_name.strip(),
            "sample_count": len(vectors),
            "meta": meta
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/profiles/compare")
async def compare_against_profile(
    target_speaker: str = Form(...),
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Compares an incoming audio file against a selected target person's enrolled dataset profile.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    profile_vec = get_profile_vector(target_speaker)
    if profile_vec is None:
        raise HTTPException(status_code=404, detail=f"Voice profile for '{target_speaker}' not found.")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # 1. Deepfake forensic scan
        forensic_res = authenticator.analyze(temp_path)

        # 2. Extract feature vector of test file
        test_vec = extract_speaker_embedding(temp_path)

        if test_vec is not None and profile_vec is not None:
            euc_dist = float(norm(test_vec - profile_vec))
            rbf_score = np.exp(-(euc_dist ** 2) / (2 * (1.35 ** 2))) * 100.0
            acoustic_similarity = round(float(max(0.0, min(100.0, rbf_score))), 2)
        else:
            acoustic_similarity = 0.0

        # Verdict logic
        if forensic_res["verdict"] == "FRAUD":
            verdict_text = f"DEEPFAKE / SYNTHETIC VOICE (Impersonating {target_speaker})"
            is_match = False
        elif acoustic_similarity >= 68.0:
            verdict_text = f"AUTHENTIC MATCH ({target_speaker})"
            is_match = True
        elif acoustic_similarity >= 45.0:
            verdict_text = f"INCONCLUSIVE / POSSIBLE MATCH ({target_speaker})"
            is_match = False
        else:
            verdict_text = f"DIFFERENT VOICE (Does not match {target_speaker})"
            is_match = False

        return {
            "target_speaker": target_speaker,
            "filename": file.filename,
            "verdict": verdict_text,
            "is_speaker_match": is_match,
            "target_profile_similarity": acoustic_similarity,
            "deepfake_analysis": forensic_res
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile comparison failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/dataset/upload")
async def upload_custom_dataset(
    category: str = Form("REAL"), # 'REAL' or 'FAKE'
    files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Allows users to upload MULTIPLE custom audio files or ZIP datasets directly to expand the model training dataset.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided.")

    target_dir = KAGGLE_REAL_DIR if category.upper() == "REAL" else (KAGGLE_REAL_DIR.parent / "FAKE")
    target_dir.mkdir(parents=True, exist_ok=True)

    added_files = 0
    temp_dir = tempfile.mkdtemp()

    try:
        for idx, file_item in enumerate(files):
            if not file_item.filename:
                continue

            suffix = os.path.splitext(file_item.filename)[1].lower()
            temp_path = os.path.join(temp_dir, f"upload_{idx}{suffix}")

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file_item.file, buffer)

            if suffix == ".zip":
                with zipfile.ZipFile(temp_path, "r") as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
                            zip_ref.extract(zip_info, target_dir)
                            added_files += 1
            elif suffix in [".wav", ".mp3", ".flac", ".ogg"]:
                dest_file = target_dir / file_item.filename
                shutil.copy(temp_path, dest_file)
                added_files += 1

        return {
            "status": "success",
            "category": category.upper(),
            "added_files_count": added_files,
            "message": f"Successfully added {added_files} audio sample(s) to user local dataset ({category.upper()})!"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset import failed: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
