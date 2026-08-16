# VoiceGuard AI

VoiceGuard AI is a real-time voice forensic and deepfake detection platform. It combines acoustic feature extraction, Gaussian Mixture Models (GMM), and machine learning classifiers to analyze audio samples, verify speaker authenticity, and detect synthetic or cloned voices.

## Key Features

- **Deepfake Audio Detection**: Analyzes audio spectrums and acoustic feature distributions (MFCCs, Spectral Centroid, ZCR, Chroma) to distinguish genuine speech from AI-generated clones.
- **Speaker Verification & Comparison (1:1 & 1:N)**: Enrolls speaker voice profiles and compares target audio against reference profiles using log-likelihood ratio (LLR) scoring.
- **Real-Time Visual Forensics**: Displays waveform visualizations, probability breakdowns, and confidence indicators via an interactive web dashboard.
- **Profile Management**: Enroll, list, and manage enrolled speaker acoustic profiles.

## Architecture & Technology Stack

- **Frontend**: React (Vite), TailwindCSS, Lucide Icons, HTML5 Audio API
- **Backend API**: Python, Flask, Flask-CORS
- **Machine Learning & Signal Processing**:
  - `librosa` / `scipy`: Audio feature extraction and signal preprocessing
  - `scikit-learn`: Support Vector Machines (SVM) & Gaussian Mixture Models (GMM)
  - `PyTorch`: Deep neural network model pipeline

## Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ and Node.js 18+ installed on your system.

### 2. Environment Setup

#### Backend Setup
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

#### Frontend Setup
```bash
npm install
```

### 3. Running the Application

You can start both backend and frontend servers using the helper script:
```bash
python start_servers.py
```
Or start them independently:
- **Backend API**: `python backend/app.py` (runs on `http://localhost:5000`)
- **Frontend App**: `npm run dev` (runs on `http://localhost:5173`)

## Project Structure

```
voiceguard_ai/
├── backend/
│   ├── app.py                     # Main Flask REST API service
│   ├── compare_audio_files.py     # Standalone audio comparison utility
│   ├── enrolled_profiles/         # Enrolled speaker profile metadata & embeddings
│   ├── ml_system/                 # Machine learning pipeline modules
│   │   ├── features/              # Feature extraction (MFCCs, Spectral, etc.)
│   │   ├── inference/             # Model evaluation & prediction handlers
│   │   ├── models/                # Saved model weights (.pkl / .pth)
│   │   └── training/              # Model training scripts
│   └── routes/                    # API route handlers (predict, compare, profiles)
├── src/                           # React frontend application
│   ├── components/                # UI components (AudioComparison, ProfileManager, Result)
│   ├── effects/                   # Interactive visual effects
│   └── App.jsx                    # Core application layout
├── start_servers.py               # Server startup utility
└── README.md
```