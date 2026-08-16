# VOICEGUARD AI - VIVA DEFENSE & PRESENTATION DEMO GUIDE

---

## 1. Project Synopsis (What to Say in the 1st Minute)
> *"Good morning respected External Examiner and Supervisor. Our major project is titled **VoiceGuard AI: Scaling The Voice Similarity**. It is a biometric audio forensic system designed to protect against AI voice cloning fraud and vishing scams. It performs 1:1 dual-voice similarity scaling, 1:N multi-speaker dataset identification, and single-audio synthetic deepfake scanning using 65-dimensional acoustic feature extraction and 16-component Gaussian Mixture Models."*

---

## 2. Top Viva Questions & Answers

### Q1: Why use Gaussian Mixture Models (GMM) instead of a simple Deep Neural Network?
**Answer:**
GMMs offer strong mathematical interpretability, fast training/inference (< 1 second), low hardware overhead, and exceptional performance on short reference audio clips (3-5 seconds). Unlike black-box neural networks, GMM log-likelihood scores provide an exact acoustic distance measure that can be mathematically mapped to a 0–100% similarity score.

---

### Q2: What features are extracted from the audio signals?
**Answer:**
We extract a **65-dimensional feature vector** for every frame:
1. **20 MFCCs (Mel-Frequency Cepstral Coefficients)**: Captures vocal tract geometry & timbre.
2. **20 Delta MFCCs**: Captures velocity (first derivative) of spectral change.
3. **20 Delta-Delta MFCCs**: Captures acceleration (second derivative) of spectral change.
4. **Spectral Dynamics**: Spectral Centroid, Spectral Flatness, Spectral Crest Factor.
5. **Non-Linear Entropy**: Shannon Entropy & Renyi Entropy to measure synthetic signal randomness.

---

### Q3: Why is Cepstral Mean Normalization (CMN) applied?
**Answer:**
CMN subtracts the temporal mean of the feature vector across all frames ($y_t = x_t - \bar{x}$). This removes channel-induced acoustic distortions caused by different microphones, telephone lines, or recording hardware, ensuring baseline invariance.

---

### Q4: Why is audio resampled to 8000 Hz with a 300–3400 Hz Bandpass Filter?
**Answer:**
We conform to the **ITU-T G.712 telephony standard**. Telephone networks limit audio bandwidth to 300 Hz – 3400 Hz. Standardizing audio to 8 kHz sampling rate ensures VoiceGuard AI works effectively on telephone calls and cellular recordings.

---

### Q5: How is the Similarity Score (0% – 100%) calculated?
**Answer:**
The log-likelihood score $\mathcal{L} = \log p(Y_{\text{suspect}} \mid \lambda_{\text{ref}})$ measures acoustic similarity. We apply an exponential distance transformation:
$$S = 100 \times \exp\left(-\frac{|\mathcal{L}_{\text{suspect}} - \mathcal{L}_{\text{ref}}|}{\tau}\right)$$
- Score $\ge 75\%$: **AUTHENTIC** (Low Risk)
- Score $45\% - 74\%$: **INCONCLUSIVE** (Moderate Risk)
- Score $< 45\%$: **FRAUDULENT / SYNTHETIC** (High Risk)

---

## 3. Step-by-Step Live Presentation Demo Script

### Step 1: Launch Local Services
1. **Backend Service**:
   ```powershell
   .\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000
   ```
2. **Frontend UI**:
   ```powershell
   npm run dev
   ```
   Open browser at: `http://localhost:5173`

---

### Step 2: Perform 1:1 Pairwise Similarity Demo
1. Click **`🎙️ Dual-Voice Similarity Engine (1:1)`**.
2. Drag & drop `test_samples/Speaker_Alice_Reference.wav` as Reference Audio.
3. Drag & drop `test_samples/Speaker_Alice_Authentic_Sample.wav` as Suspect Audio.
4. Click **Execute 1:1 GMM Similarity Scaling**.
5. *Result*: Shows High Similarity Percentage (~85% - 95%) & **AUTHENTIC** verdict.

---

### Step 3: Perform 1:N Multi-Speaker Dataset Identification Demo
1. Click **`📚 Multi-Speaker Dataset Identification (1:N)`**.
2. Under Dataset Candidates, upload `Speaker_Alice_Reference.wav`, `Speaker_Bob_Enrolled.wav`, and `Speaker_Charlie_Enrolled.wav`.
3. Under Target Suspect Audio, upload `Speaker_Alice_Authentic_Sample.wav`.
4. Click **Run 1:N Multi-Speaker Dataset Identification**.
5. *Result*: Leaderboard ranks `Speaker Alice` at #1 with Top Match Score!

---

### Step 4: Perform Single Audio Deepfake Scanner Demo
1. Click **`🔍 Single Audio Deepfake Scanner`**.
2. Upload `test_samples/Suspect_Deepfake_Clone.wav`.
3. Click **Analyze Audio Authenticity**.
4. *Result*: System flags file as **FRAUDULENT / SYNTHETIC** due to static entropy variance.

---
*Document prepared for B.Tech CSE Major Project Final Year Viva Defense (Group 19, UIT Prayagraj).*
