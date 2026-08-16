import sys
import os

# Add backend to path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.evaluator import evaluator

def run_test():
    # Test cases: (log_likelihood, expected_verdict_prefix)
    test_cases = [
        (-80.0, "GENUINE"),   # High similarity
        (-950.0, "FAKE"),     # Low similarity
        (-400.0, "GENUINE"),  # Above 60% threshold (~63%)
        (-500.0, "FAKE")      # Below 60% threshold (~52%)
    ]

    print("Starting Forensic Evaluation Verification...")
    print("-" * 40)

    for i, (ll, expected) in enumerate(test_cases):
        result = evaluator.evaluate(ll)
        
        print(f"Test Case {i+1}: Log-Likelihood = {ll}")
        print(f"Similarity Score:  {result['similarity_score']}%")
        print(f"Fraud Probability: {result['fraud_probability']}")
        print(f"Verdict:           {result['verdict']}")
        
        if result['verdict'] == expected:
            print(f"Calibration:       PASSED (Expected {expected})")
        else:
            print(f"Calibration:       FAILED (Expected {expected}, Got {result['verdict']})")
        print("-" * 40)

if __name__ == "__main__":
    run_test()
