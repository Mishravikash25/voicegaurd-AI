import sys
import os
import numpy as np

# Add backend to path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.model import forensic_model

def generate_synthetic_features(n_frames=500, n_dims=44, seed=42):
    """Generates synthetic forensic features."""
    np.random.seed(seed)
    # Distribution 1: "Genuine" Voice Pattern
    mean = np.linspace(1, 10, n_dims)
    genuine_features = np.random.normal(loc=mean, scale=2.0, size=(n_frames, n_dims))
    
    # Distribution 2: "Fraudulent/Noise" Pattern
    fake_features = np.random.uniform(low=-20, high=20, size=(n_frames, n_dims))
    
    return genuine_features, fake_features

def run_test():
    print("Generating synthetic forensic data...")
    genuine, fake = generate_synthetic_features()
    
    try:
        # 1. Training
        print("\nStep 1: Training Forensic Core...")
        forensic_model.train(genuine)
        
        # 2. Saving
        print("\nStep 2: Persisting Model...")
        forensic_model.save_model()
        
        # 3. Loading
        print("\nStep 3: Re-Importing Model...")
        # Clear model first
        forensic_model.model = None
        success = forensic_model.load_model()
        if not success:
            raise Exception("Model load failed.")

        # 4. Scoring
        print("\nStep 4: Forensic Analysis Scoring...")
        genuine_score = forensic_model.get_log_likelihood(genuine)
        fake_score = forensic_model.get_log_likelihood(fake)
        
        print(f"Genuine Sample Score (Log-Likelihood): {genuine_score:.4f}")
        print(f"Fake Sample Score (Log-Likelihood):    {fake_score:.4f}")
        
        # 5. Probability Check
        genuine_prob = forensic_model.calculate_fraud_probability(genuine_score)
        fake_prob = forensic_model.calculate_fraud_probability(fake_score)
        
        print(f"\nFraud Probability (Genuine Sample): {genuine_prob:.2f}%")
        print(f"Fraud Probability (Fake Sample):    {fake_prob:.2f}%")
        
        if genuine_prob < 50 and fake_prob > 50:
            print("\nForensic Calibration Check: PASSED")
        else:
            print("\nForensic Calibration Check: FAILED")

    finally:
        # Cleanup
        if os.path.exists(forensic_model.model_path):
            # os.remove(forensic_model.model_path)
            # print(f"Cleaned up {forensic_model.model_path}")
            pass

if __name__ == "__main__":
    run_test()
