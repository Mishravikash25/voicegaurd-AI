from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
import tempfile
import sys
from typing import Dict, Any

# Ensure we can import from ml_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_system.inference.predict import authenticator

router = APIRouter()

@router.post("/predict")
async def predict_voice(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Forensic Endpoint: Analyzes an uploaded audio file for voice authenticity.
    Uses the new advanced ML System (SVM + Feature Scaling).
    """
    # 1. Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not (file.filename.endswith('.wav') or file.filename.endswith('.mp3') or file.filename.endswith('.mp4')):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, and .mp4 files are supported")

    # 2. Save temporary file for the ML system to process
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        # 3. Analyze using the new ML System Pipeline
        # Returned format: {"verdict": "...", "fraud_probability": X.XX, "similarity_score": X.XX}
        results = authenticator.analyze(temp_path)
        
        return results

    except Exception as e:
        print(f"Forensic Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # 4. Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
