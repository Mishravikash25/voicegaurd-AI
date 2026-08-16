import os
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional

class ForensicModel:
    def __init__(self, n_components: int = 16, covariance_type: str = 'diag'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model: Optional[GaussianMixture] = None
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.model_path = os.path.join(self.model_dir, 'gmm_model.pkl')

        # Ensure models directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train(self, features: np.ndarray):
        """Trains a GMM on the provided feature matrix."""
        # features shape: (Time Frames, Dimensions)
        print(f"Training GMM with {self.n_components} components...")
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            init_params='kmeans',
            max_iter=200,
            n_init=3,
            random_state=42
        )
        self.model.fit(features)
        print("Training complete.")

    def save_model(self):
        """Serializes the trained model to disk."""
        if self.model is None:
            raise ValueError("No model trained to save.")
        
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Loads a trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        print(f"No model found at {self.model_path}")
        return False

    def get_log_likelihood(self, features: np.ndarray) -> float:
        """Computes the average log-likelihood of the features given the model."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model must be trained or loaded before scoring.")
        
        # score_samples returns log-likelihood per sample (frame)
        scores = self.model.score_samples(features)
        return float(np.mean(scores))

    def calculate_fraud_probability(self, score: float, threshold: float = -150.0) -> float:
        """
        Heuristic to convert log-likelihood to a 0-1 probability.
        Note: In a real system, this would be based on a calibrated EER threshold.
        """
        # Lower score (more negative) means more likely fraud
        # Example sigmoid-like mapping
        # If score is close to threshold, probability is ~0.5
        # Higher score -> Lower fraud probability -> More genuine
        # We'll return 1 - sigmoid(score - threshold)
        diff = score - threshold
        prob = 1 / (1 + np.exp(diff))
        return float(np.clip(prob * 100, 0, 100))

# Reusable instance
forensic_model = ForensicModel()
