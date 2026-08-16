# VoiceGuard-AI: Multi-User & DeepFake Dataset Training Guide

This guide provides a comprehensive walkthrough for preparing dataset files, integrating the Kaggle **DEEP-VOICE** dataset (`birdy654/deep-voice-deepfake-voice-recognition`), and training **VoiceGuard-AI** models for voice recognition and deepfake detection.

---

## 1. Prerequisites & Environment Setup

Ensure you are in the `backend` directory of the repository and install all required dependencies (including `kagglehub` for downloading Kaggle datasets directly):

```bash
cd voicegaurd-AI-main/backend
pip install -r requirements.txt
pip install kagglehub[pandas-datasets]
```

---

## 2. Loading the Kaggle Dataset (`birdy654/deep-voice-deepfake-voice-recognition`)

The **DEEP-VOICE** dataset contains real speech and RVC AI-generated deepfake voice conversions. You can load pre-extracted CSV feature tables or raw audio datasets directly using `kagglehub`:

### Python Snippet for Loading Kaggle Dataset

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the file path within the dataset (e.g., "DATASET-balanced.csv")
file_path = "DATASET-balanced.csv"

# Load the dataset using pandas adapter
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS
    "birdy654/deep-voice-deepfake-voice-recognition",
    file_path,
    # Provide any additional arguments like sql_query or pandas_kwargs.
)

print("First 5 records:", df.head())
```

---

## 3. Custom Multi-User Dataset Directory Structure

If you are training on your own multi-speaker audio recordings, structure your local audio files as follows:

```text
dataset/
├── user_001/
│   ├── sample_01.wav
│   ├── sample_02.wav
│   └── sample_03.wav
├── user_002/
│   ├── sample_01.wav
│   └── sample_02.wav
└── user_003/
    ├── sample_01.wav
    └── sample_02.wav
```

### Dataset Recommendations
* **Clip Length:** 1 to 3 seconds per `.wav` chunk.
* **Speaker Audio:** At least 15–30 seconds of speech per speaker.
* **Audio Format:** Mono 16 kHz or 22.05 kHz `.wav` files.

---

## 4. Handling Mixed Audio (Multiple Speakers in One File)

For long continuous recordings with multiple speakers:

1. **Voice Activity Detection (VAD) & Speaker Diarization:**
   * Use tools such as `pyannote.audio` or `silero-vad` to identify speech boundaries and separate speakers.
2. **Export to Class Folders:**
   * Slice audio into 2-second clips based on timestamp intervals.
   * Save each speaker's clips into their respective target folder (`user_001/`, `user_002/`, etc.).

---

## 5. Preprocessing & Feature Extraction Pipeline

VoiceGuard-AI processes audio through `backend/ml_system/features/`:

1. **Preprocessing (`preprocess.py`):**
   * Normalizes audio amplitude.
   * Trims silences and resamples to standard sample rates.
2. **Feature Extraction (`extract.py` / `extractor.py`):**
   * Extracts acoustic representations:
     * **MFCCs** (Mel-Frequency Cepstral Coefficients)
     * **Spectral Centroid & Bandwidth**
     * **Zero-Crossing Rate (ZCR)**
     * **Chroma Features**

---

## 6. Training Strategies

VoiceGuard-AI supports two main training strategies depending on your task:

### Strategy A: Gaussian Mixture Models (GMM)
* **Best for:** Fast training on smaller datasets, individual user profiles.
* **Command:**
  ```bash
  python ml_system/training/train.py --data_dir /path/to/dataset
  ```

### Strategy B: Deep Learning Classifier (CNN / MLP)
* **Best for:** DeepFake voice classification or multi-speaker identification on larger datasets.
* **Command:**
  ```bash
  python ml_system/training/train_dl.py --data_dir /path/to/dataset
  ```

---

## 7. Model Evaluation & Real-Time Inference

1. **Generate Plots & Confusion Matrices:**
   ```bash
   python ml_system/training/visualize.py
   ```
2. **Test Scripts:**
   ```bash
   python scripts/test_model.py
   python scripts/test_features.py
   ```
3. **Deployment:**
   * Model checkpoints are stored in `backend/models/`.
   * Real-time API endpoints in `backend/app.py` load these checkpoints via `backend/routes/predict.py`.
