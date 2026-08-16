import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Union

# Ensure we can import from the ml_system root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.preprocess import preprocessor
from features.extract import extractor
from models.model_manager import model_manager

# Configure logging
logger = logging.getLogger(__name__)

class VoiceAuthenticator:
    """
    End-to-End inference pipeline for VoiceGuard AI.
    Handles a raw audio file and outputs a forensic verdict.
    """
    def __init__(self):
        # We trigger model load here. ModelManager ensures this is fast and only happens once.
        try:
            self.model, self.scaler = model_manager.load_model(
                model_name="best_svm_model.pkl", 
                scaler_name="feature_scaler.pkl"
            )
        except FileNotFoundError:
            logger.warning("Models not found. Inference will fail unless model is trained.")
            self.model, self.scaler = None, None

    def analyze(self, audio_path: str) -> Dict[str, Union[float, str]]:
        """
        Analyzes an audio file and returns forensic metrics.
        
        Outputs:
        - verdict: 'GENUINE' or 'FRAUD'
        - fraud_probability: float (0.00 to 100.00)
        - similarity_score: float (0.00 to 100.00)
        """
        if not self.model or not self.scaler:
            raise RuntimeError("Model or Scaler not loaded. Cannot perform inference.")

        logger.info(f"Analyzing audio: {audio_path}")
        
        # 1. Preprocessing
        # This will load, resample (16kHz), filter (300-3400Hz), trim, and normalize
        cleaned_audio = preprocessor.process(audio_path)
        
        if cleaned_audio is None or len(cleaned_audio) == 0:
            raise ValueError(f"Provided audio file '{audio_path}' is empty or entirely silent.")

        # 2. Extract Features
        # Temporarily disable extractor's internal scalar since we use the globally fitted scaler
        extractor.scaler = None 
        raw_feature_vector = extractor.extract_features(cleaned_audio)
        
        if raw_feature_vector is None:
            raise RuntimeError("Failed to extract acoustic features.")

        # 3. Global Scaling
        # Scaler expects a 2D array [samples, features]. Since we have 1 sample, we reshape to (1, -1)
        scaled_features = self.scaler.transform(raw_feature_vector.reshape(1, -1))

        # 4. Predict
        # model.predict_proba returns probability for [Genuine, Fake]
        # Our labels during training: 0 = Genuine, 1 = Fake
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        genuine_prob = float(probabilities[0] * 100)
        fake_prob = float(probabilities[1] * 100)
        
        # 5. Interpret Metrics
        # Verdict: FAKE if Fake probability > 50%, else GENUINE
        verdict = "FRAUD" if fake_prob > 50.0 else "GENUINE"
        
        # Similarity Score correlates directly with Genuine Probability
        similarity_score = genuine_prob

        results = {
            "verdict": verdict,
            "fraud_probability": round(fake_prob, 2),
            "similarity_score": round(similarity_score, 2)
        }

        logger.info(f"Verdict: {results['verdict']} (Similarity: {results['similarity_score']}%)")
        return results

# Reusable singleton instance for API usage
authenticator = VoiceAuthenticator()

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="VoiceGuard AI Forensic Analysis tool")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (.wav, .mp3) to analyze")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        logger.error(f"File not found: {args.audio_file}")
        sys.exit(1)

    try:
        authenticator = VoiceAuthenticator()
        results = authenticator.analyze(args.audio_file)
        
        print("\n" + "="*40)
        print("🔍 VOICEGUARD FORENSIC REPORT")
        print("="*40)
        print(f"File: {os.path.basename(args.audio_file)}")
        print(f"Verdict:             {results['verdict']}")
        print(f"Similarity Score:    {results['similarity_score']}%")
        print(f"Fraud Probability:   {results['fraud_probability']}%")
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
