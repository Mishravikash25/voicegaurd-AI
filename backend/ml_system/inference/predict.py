import os
import sys
import logging
import numpy as np
import torch
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
        try:
            self.model, self.scaler = model_manager.load_model(
                model_name="best_svm_model.pkl", 
                scaler_name="feature_scaler.pkl"
            )
        except FileNotFoundError:
            logger.warning("SVM Model or Scaler not found.")
            self.model, self.scaler = None, None

        # Load PyTorch CNN model if available
        self.cnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_pytorch_model.pth")
        if os.path.exists(cnn_path):
            try:
                from models.deep_learning import SpectrogramCNN
                self.cnn_model = SpectrogramCNN().to(self.device)
                self.cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
                self.cnn_model.eval()
                logger.info(f"Loaded PyTorch Spectrogram CNN on {self.device}")
            except Exception as e:
                logger.warning(f"Could not load PyTorch CNN model: {e}")

    def analyze(self, audio_path: str) -> Dict[str, Union[float, str]]:
        """
        Analyzes an audio file using 2-second sliding windows with CNN + SVM ensemble.
        Outputs:
        - verdict: 'GENUINE' or 'FRAUD'
        - fraud_probability: float (0.00 to 100.00)
        - similarity_score: float (0.00 to 100.00)
        """
        if not self.model and not self.cnn_model:
            raise RuntimeError("No models loaded. Cannot perform inference.")

        logger.info(f"Analyzing audio: {audio_path}")
        
        cleaned_audio = preprocessor.load_and_resample(audio_path)
        if cleaned_audio is None or len(cleaned_audio) == 0:
            raise ValueError(f"Provided audio file '{audio_path}' is empty or entirely silent.")

        sr = 16000
        chunk_len = 2 * sr  # 2-second window
        step = chunk_len // 2  # 50% overlap

        # Handle short audio by padding if less than 2 seconds
        if len(cleaned_audio) < chunk_len:
            pad_len = chunk_len - len(cleaned_audio)
            audio_windows = [np.pad(cleaned_audio, (0, pad_len), mode='constant')]
        else:
            audio_windows = [
                cleaned_audio[i:i + chunk_len]
                for i in range(0, len(cleaned_audio) - chunk_len + 1, step)
            ]
            if len(audio_windows) == 0:
                audio_windows = [np.pad(cleaned_audio, (0, chunk_len - len(cleaned_audio)), mode='constant')]

        cnn_fake_probs = []
        svm_fake_probs = []

        # 1. PyTorch CNN Sliding Window Analysis
        if self.cnn_model is not None:
            with torch.no_grad():
                for chunk in audio_windows:
                    spec = extractor.extract_mel_spectrogram(chunk)
                    tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                    out = self.cnn_model(tensor_spec)
                    prob = torch.softmax(out, dim=1)[0][1].item() * 100
                    cnn_fake_probs.append(prob)

        # 2. SVM Sliding Window Analysis
        if self.model is not None and self.scaler is not None:
            for chunk in audio_windows:
                raw_feature_vector = extractor.extract_features_v2(chunk)
                if raw_feature_vector is not None:
                    scaled_features = self.scaler.transform(raw_feature_vector.reshape(1, -1))
                    prob = self.model.predict_proba(scaled_features)[0][1] * 100
                    svm_fake_probs.append(prob)

        # 3. Decision Aggregation via Window Means
        if len(cnn_fake_probs) > 0:
            avg_cnn_fake = float(np.mean(cnn_fake_probs))
        else:
            avg_cnn_fake = 0.0

        if len(svm_fake_probs) > 0:
            avg_svm_fake = float(np.mean(svm_fake_probs))
        else:
            avg_svm_fake = 0.0

        # Weighted Ensemble: 60% PyTorch Spectrogram CNN + 40% SVM
        if self.cnn_model is not None and self.model is not None:
            final_fake_prob = 0.6 * avg_cnn_fake + 0.4 * avg_svm_fake
        elif self.cnn_model is not None:
            final_fake_prob = avg_cnn_fake
        else:
            final_fake_prob = avg_svm_fake

        final_fake_prob = min(max(final_fake_prob, 0.0), 100.0)
        verdict = "FRAUD" if final_fake_prob >= 50.0 else "GENUINE"
        similarity_score = 100.0 - final_fake_prob

        results = {
            "verdict": verdict,
            "fraud_probability": round(final_fake_prob, 2),
            "similarity_score": round(similarity_score, 2),
            "engine": "Spectrogram-CNN + SVM Windowed Ensemble"
        }

        logger.info(f"Verdict: {results['verdict']} (Fraud Prob: {results['fraud_probability']}%)")
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
        print("[+] VOICEGUARD FORENSIC REPORT")
        print("="*40)
        print(f"File: {os.path.basename(args.audio_file)}")
        print(f"Verdict:             {results['verdict']}")
        print(f"Similarity Score:    {results['similarity_score']}%")
        print(f"Fraud Probability:   {results['fraud_probability']}%")
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
