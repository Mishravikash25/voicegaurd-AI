"""
VoiceGuard AI: Balanced Retraining Script (v2)
================================================
Generates synthetic "fake" audio from existing REAL samples using DSP
transforms, ensures exact class balance, and retrains the SVM model.
"""

import os
import sys
import logging
import shutil
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.ndimage import median_filter, uniform_filter1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent
ML_SYS_DIR = BACKEND_DIR / "ml_system"
sys.path.insert(0, str(ML_SYS_DIR))
sys.path.insert(0, str(BACKEND_DIR))

SAMPLE_RATE = 16000


def apply_deepfake_transform_v1(audio, sr=SAMPLE_RATE):
    """Pitch-shift + vocoder smoothing: mimics voice conversion artifacts."""
    shift = np.random.choice([-3, -2, 2, 3])
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
    S = librosa.stft(shifted)
    S_mag = np.abs(S)
    S_smooth = median_filter(S_mag, size=(5, 1))
    S_reconstructed = S_smooth * np.exp(1j * np.angle(S))
    return librosa.istft(S_reconstructed, length=len(shifted))


def apply_deepfake_transform_v2(audio, sr=SAMPLE_RATE):
    """Time-stretch + band-limit: mimics neural TTS bandwidth artifacts."""
    rate = np.random.choice([0.88, 0.92, 1.08, 1.12])
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) > len(audio):
        stretched = stretched[:len(audio)]
    else:
        stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
    nyquist = 0.5 * sr
    cutoff = 4000.0 / nyquist
    b, a = scipy_signal.butter(6, cutoff, btype='low')
    filtered = scipy_signal.filtfilt(b, a, stretched)
    noise = np.random.randn(len(filtered)) * 0.005
    return filtered + noise


def apply_deepfake_transform_v3(audio, sr=SAMPLE_RATE):
    """Phase randomization: destroys natural phase coherence."""
    S = librosa.stft(audio)
    S_mag = np.abs(S)
    random_phase = np.random.uniform(-np.pi, np.pi, S.shape)
    blended_phase = 0.7 * random_phase + 0.3 * np.angle(S)
    S_fake = S_mag * np.exp(1j * blended_phase)
    result = librosa.istft(S_fake, length=len(audio))
    shift = np.random.choice([-1.5, -1, 1, 1.5])
    return librosa.effects.pitch_shift(result, sr=sr, n_steps=shift)


def apply_deepfake_transform_v4(audio, sr=SAMPLE_RATE):
    """Resampling artifacts: mimics low-quality clones."""
    low_sr = 8000
    downsampled = librosa.resample(audio, orig_sr=sr, target_sr=low_sr)
    upsampled = librosa.resample(downsampled, orig_sr=low_sr, target_sr=sr)
    if len(upsampled) > len(audio):
        upsampled = upsampled[:len(audio)]
    else:
        upsampled = np.pad(upsampled, (0, max(0, len(audio) - len(upsampled))))
    shift = np.random.choice([-2, -1, 1, 2])
    result = librosa.effects.pitch_shift(upsampled, sr=sr, n_steps=shift)
    noise = np.random.randn(len(result)) * 0.008
    return result + noise


def apply_deepfake_transform_v5(audio, sr=SAMPLE_RATE):
    """Harmonic manipulation: alters harmonic structure like voice conversion."""
    harmonic, percussive = librosa.effects.hpss(audio)
    shift = np.random.choice([-4, -3, 3, 4])
    shifted_harmonic = librosa.effects.pitch_shift(harmonic, sr=sr, n_steps=shift)
    result = 0.85 * shifted_harmonic + 0.15 * percussive
    result = uniform_filter1d(result, size=3)
    return result


TRANSFORMS = [
    apply_deepfake_transform_v1,
    apply_deepfake_transform_v2,
    apply_deepfake_transform_v3,
    apply_deepfake_transform_v4,
    apply_deepfake_transform_v5,
]


def generate_augmented_real(audio, sr=SAMPLE_RATE):
    """Generate augmented REAL samples with natural variations."""
    augmentations = []
    
    # 1. Speed perturbation (natural speech rate variation)
    for rate in [0.95, 1.05]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        augmentations.append(stretched)
    
    # 2. Volume variation (natural dynamic range)
    for gain in [0.7, 1.3]:
        augmentations.append(audio * gain)
    
    # 3. Very slight background noise (room noise)
    noise = np.random.randn(len(audio)) * 0.002
    augmentations.append(audio + noise)
    
    return augmentations


def main():
    logger.info("=" * 60)
    logger.info("VoiceGuard AI: Balanced Retraining Pipeline v2")
    logger.info("=" * 60)
    
    # 1. Locate REAL audio
    real_dir = BACKEND_DIR / "dataset" / "KAGGLE" / "AUDIO" / "REAL"
    fake_dir = BACKEND_DIR / "dataset" / "KAGGLE" / "AUDIO" / "FAKE"
    
    if not real_dir.exists() or not list(real_dir.glob("*.wav")):
        logger.error(f"No REAL audio found in {real_dir}.")
        sys.exit(1)
    
    # 2. Clean up any previous fake samples
    if fake_dir.exists():
        shutil.rmtree(fake_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Load all real audio
    real_files = sorted(list(real_dir.glob("*.wav")))
    logger.info(f"Found {len(real_files)} REAL audio files")
    
    real_audios = []
    for f in real_files:
        try:
            audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            if len(audio) >= SAMPLE_RATE:
                real_audios.append((f.name, audio))
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")
    
    logger.info(f"Loaded {len(real_audios)} valid REAL audio clips")
    
    # 4. Generate augmented REAL data to balance the dataset
    augmented_real_dir = BACKEND_DIR / "dataset" / "KAGGLE" / "AUDIO" / "REAL_AUG"
    if augmented_real_dir.exists():
        shutil.rmtree(augmented_real_dir)
    
    # We will NOT use a separate REAL_AUG folder - instead add to REAL
    real_aug_count = 0
    for name, audio in real_audios:
        augs = generate_augmented_real(audio, SAMPLE_RATE)
        for i, aug_audio in enumerate(augs):
            max_amp = np.max(np.abs(aug_audio))
            if max_amp > 0:
                aug_audio = aug_audio / max_amp * 0.95
            out_name = f"aug_{name.replace('.wav', '')}_r{i}.wav"
            sf.write(str(real_dir / out_name), aug_audio, SAMPLE_RATE)
            real_aug_count += 1
    
    total_real = len(list(real_dir.glob("*.wav")))
    logger.info(f"Augmented REAL count: {total_real} ({real_aug_count} augmented + {len(real_audios)} original)")
    
    # 5. Generate FAKE samples (match the REAL count for perfect balance)
    target_fake = total_real
    fake_per_real = max(1, target_fake // len(real_audios)) + 1
    
    logger.info(f"Generating {target_fake}+ FAKE samples ({fake_per_real} per REAL file)...")
    
    fake_count = 0
    for name, audio in real_audios:
        for i in range(fake_per_real):
            if fake_count >= target_fake:
                break
            transform = TRANSFORMS[i % len(TRANSFORMS)]
            try:
                fake_audio = transform(audio)
                max_amp = np.max(np.abs(fake_audio))
                if max_amp > 0:
                    fake_audio = fake_audio / max_amp * 0.95
                out_name = f"fake_{name.replace('.wav', '')}_v{i}.wav"
                sf.write(str(fake_dir / out_name), fake_audio, SAMPLE_RATE)
                fake_count += 1
            except Exception as e:
                logger.warning(f"Transform {i} failed for {name}: {e}")
        if fake_count >= target_fake:
            break
    
    actual_real = len(list(real_dir.glob("*.wav")))
    actual_fake = len(list(fake_dir.glob("*.wav")))
    logger.info(f"Final dataset: {actual_real} REAL, {actual_fake} FAKE")
    
    # 6. Prepare features using the training pipeline
    # Import here to avoid circular loading issues
    from ml_system.features.preprocess import preprocessor
    from ml_system.features.extract import extractor
    from ml_system.config import MODELS_DIR
    
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
    import joblib
    
    # Extract features manually (not using prepare_dataset to avoid data_loader 
    # double-counting issue with 'real' and 'REAL' folders)
    X_features = []
    y_labels = []
    
    logger.info("Extracting features from REAL samples...")
    for f in sorted(real_dir.glob("*.wav")):
        try:
            audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, duration=5, mono=True)
            cleaned = preprocessor.apply_bandpass_filter(audio)
            cleaned = preprocessor.remove_silence(cleaned)
            if cleaned is None or len(cleaned) == 0:
                continue
            cleaned = preprocessor.apply_cmn(cleaned)
            cleaned = preprocessor.normalize_amplitude(cleaned)
            
            extractor.scaler = None
            fv = extractor.extract_features(cleaned)
            if fv is not None:
                X_features.append(fv)
                y_labels.append(0)  # Genuine
        except Exception as e:
            logger.warning(f"Skipping REAL {f.name}: {e}")
    
    real_feature_count = len(X_features)
    
    logger.info("Extracting features from FAKE samples...")
    for f in sorted(fake_dir.glob("*.wav")):
        try:
            audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, duration=5, mono=True)
            cleaned = preprocessor.apply_bandpass_filter(audio)
            cleaned = preprocessor.remove_silence(cleaned)
            if cleaned is None or len(cleaned) == 0:
                continue
            cleaned = preprocessor.apply_cmn(cleaned)
            cleaned = preprocessor.normalize_amplitude(cleaned)
            
            extractor.scaler = None
            fv = extractor.extract_features(cleaned)
            if fv is not None:
                X_features.append(fv)
                y_labels.append(1)  # Fake
        except Exception as e:
            logger.warning(f"Skipping FAKE {f.name}: {e}")
    
    X = np.array(X_features)
    y = np.array(y_labels)
    
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Features extracted: {len(X)} samples, {X.shape[1]} dims")
    logger.info(f"Class balance: Genuine(0)={class_dist.get(0,0)}, Fake(1)={class_dist.get(1,0)}")
    
    if len(unique) < 2:
        logger.error("CRITICAL: Only one class. Need both REAL and FAKE.")
        sys.exit(1)
    
    # 7. Train with class_weight='balanced' to handle any remaining imbalance
    logger.info("Training SVM with GridSearchCV (5-Fold CV)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training: {X_train_scaled.shape}, Testing: {X_test_scaled.shape}")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    svm = SVC(probability=True, random_state=42, class_weight='balanced')
    
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info("=" * 40)
    logger.info("FINAL MODEL EVALUATION")
    logger.info("=" * 40)
    logger.info(f"Accuracy:  {accuracy * 100:.2f}%")
    logger.info(f"Confusion Matrix:")
    logger.info(f"[{conf_matrix[0][0]:3d}  {conf_matrix[0][1]:3d}]  <- Genuine")
    logger.info(f"[{conf_matrix[1][0]:3d}  {conf_matrix[1][1]:3d}]  <- Fake")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Genuine", "Fake"], zero_division=0))
    
    # 8. Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_svm_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Scaler saved to: {scaler_path}")
    
    # 9. Verification: test inference
    logger.info("")
    logger.info("=" * 40)
    logger.info("POST-TRAINING VERIFICATION")
    logger.info("=" * 40)
    
    # Reload fresh
    model = joblib.load(model_path)
    scaler_loaded = joblib.load(scaler_path)
    
    # Test ALL real files
    logger.info("\n--- REAL AUDIO (should be GENUINE) ---")
    real_originals = [f for f in real_dir.glob("*.wav") if not f.name.startswith("aug_")]
    for f in sorted(real_originals):
        audio = preprocessor.process(str(f))
        if audio is not None:
            extractor.scaler = None
            fv = extractor.extract_features(audio)
            if fv is not None:
                scaled = scaler_loaded.transform(fv.reshape(1, -1))
                proba = model.predict_proba(scaled)[0]
                verdict = "GENUINE" if proba[0] > proba[1] else "FRAUD"
                logger.info(f"  {f.name}: {verdict} (Genuine={proba[0]*100:.1f}%, Fake={proba[1]*100:.1f}%)")
    
    # Test ALL fake files
    logger.info("\n--- FAKE AUDIO (should be FRAUD) ---")
    for f in sorted(list(fake_dir.glob("*.wav"))[:6]):
        audio = preprocessor.process(str(f))
        if audio is not None:
            extractor.scaler = None
            fv = extractor.extract_features(audio)
            if fv is not None:
                scaled = scaler_loaded.transform(fv.reshape(1, -1))
                proba = model.predict_proba(scaled)[0]
                verdict = "GENUINE" if proba[0] > proba[1] else "FRAUD"
                logger.info(f"  {f.name}: {verdict} (Genuine={proba[0]*100:.1f}%, Fake={proba[1]*100:.1f}%)")
    
    # Test with random noise
    logger.info("\n--- NOISE (should be FRAUD) ---")
    np.random.seed(99)
    noise = np.random.randn(SAMPLE_RATE * 2)
    extractor.scaler = None
    fv = extractor.extract_features(noise)
    if fv is not None:
        scaled = scaler_loaded.transform(fv.reshape(1, -1))
        proba = model.predict_proba(scaled)[0]
        verdict = "GENUINE" if proba[0] > proba[1] else "FRAUD"
        logger.info(f"  Random noise: {verdict} (Genuine={proba[0]*100:.1f}%, Fake={proba[1]*100:.1f}%)")
    
    # Clean up augmented REAL files to keep dataset clean
    logger.info("\nCleaning up augmented REAL files...")
    for f in real_dir.glob("aug_*.wav"):
        f.unlink()
    
    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
