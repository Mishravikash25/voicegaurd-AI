import requests
import os
import numpy as np
import soundfile as sf

def generate_test_audio(filename="test_payload.wav", duration=1.0, sr=16000):
    """Generates a 1kHz sine wave for testing the API."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
    sf.write(filename, audio, sr)
    print(f"Generated {filename}")
    return filename

def test_predict_api():
    url = "http://localhost:8000/predict"
    test_file = generate_test_audio()
    
    try:
        print(f"Sending {test_file} to {url}...")
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "audio/wav")}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(response.json())
            print("\nAPI Integration: SUCCESS")
        else:
            print(f"Error Response: {response.text}")
            print("\nAPI Integration: FAILED")

    except Exception as e:
        print(f"Request failed: {str(e)}")

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up {test_file}")

if __name__ == "__main__":
    test_predict_api()
