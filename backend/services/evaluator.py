from typing import Dict, Any

class ForensicEvaluator:
    def __init__(self, target_threshold: float = 60.0):
        self.target_threshold = target_threshold
        # Calibration points for log-likelihood to similarity mapping
        # These would be derived from a larger validation set in production
        self.min_ll = -1000.0  # Represents ~0% similarity
        self.max_ll = -50.0    # Represents ~100% similarity

    def normalize_similarity(self, log_likelihood: float) -> float:
        """
        Maps raw log-likelihood to a 0-100 similarity score.
        Uses a simple linear interpolation with clipping.
        """
        # Linear map: (val - min) / (max - min) * 100
        score = ((log_likelihood - self.min_ll) / (self.max_ll - self.min_ll)) * 100
        return float(max(0, min(100, score)))

    def calculate_fraud_probability(self, similarity_score: float) -> float:
        """
        Derives fraud probability (0-1) from the similarity score.
        Higher similarity -> Lower fraud probability.
        """
        # Inverse relationship
        prob = (100 - similarity_score) / 100
        return float(max(0, min(1, prob)))

    def get_verdict(self, similarity_score: float) -> str:
        """Determines the forensic verdict based on the threshold."""
        if similarity_score >= self.target_threshold:
            return "GENUINE"
        return "FAKE"

    def evaluate(self, log_likelihood: float) -> Dict[str, Any]:
        """
        Performs the full evaluation and returns the standardized result payload.
        """
        similarity = self.normalize_similarity(log_likelihood)
        fraud_prob = self.calculate_fraud_probability(similarity)
        verdict = self.get_verdict(similarity)

        return {
            "similarity_score": round(similarity, 2),
            "fraud_probability": round(fraud_prob, 4),
            "verdict": verdict
        }

# Reusable instance
evaluator = ForensicEvaluator()
