"""
VoiceGuard AI — Complete Retraining Pipeline on FoR (Fake-or-Real) Dataset
===========================================================================
Trains BOTH the SVM and CNN models from scratch using the locally saved
FoR dataset at S:\\project_train.

Strategy:
  • SVM  → trained on for-norm (26k+26k balanced, variable-length, normalized audio)
           uses the 26-dim extract_features_v2() schema matching the inference pipeline
  • CNN  → trained on for-2sec + for-rerec merged (~12k+12k, fixed 2-second clips)
           uses 128-bin Log-Mel Spectrograms

Artifacts saved to:  backend/ml_system/models/
  - best_svm_model.pkl
  - feature_scaler.pkl
  - best_pytorch_model.pth

Usage:
    python retrain_for_dataset.py --model both          # Train SVM + CNN (default)
    python retrain_for_dataset.py --model svm           # SVM only
    python retrain_for_dataset.py --model cnn           # CNN only
    python retrain_for_dataset.py --model svm --limit 5000   # SVM on first 5000/class
"""

import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
import joblib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("VoiceGuard-Retrain")

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
ML_MODELS_DIR = BACKEND_DIR / "ml_system" / "models"
ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = Path(r"S:\project_train")
FOR_NORM_DIR = DATASET_ROOT / "for-norm" / "for-norm"
FOR_2SEC_DIR = DATASET_ROOT / "for-2sec" / "for-2seconds"
FOR_REREC_DIR = DATASET_ROOT / "for-rerec" / "for-rerecorded"

# Audio config (must match ml_system/config.py)
SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.mp4'}


# ──────────────────────────────────────────────────────────────────────
# Feature Extraction (SVM) — matches extract_features_v2() exactly
# ──────────────────────────────────────────────────────────────────────
def extract_features_v2(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    26-dimensional feature vector matching the inference pipeline schema:
    chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
    zero_crossing_rate, mfcc1..mfcc20
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH, n_fft=N_FFT)

    features = [
        float(np.mean(chroma)),
        float(np.mean(rms)),
        float(np.mean(centroid)),
        float(np.mean(bandwidth)),
        float(np.mean(rolloff)),
        float(np.mean(zcr)),
    ]
    features.extend(np.mean(mfccs, axis=1).tolist())  # mfcc1..mfcc20

    vec = np.array(features, dtype=float)
    vec = np.nan_to_num(vec)
    return vec


def _process_single_file_svm(args):
    """Worker function for parallel SVM feature extraction."""
    file_path, label = args
    try:
        audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        if audio is None or len(audio) < SAMPLE_RATE * 0.1:  # skip <0.1s
            return None
        vec = extract_features_v2(audio)
        return (vec, label)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Feature Extraction (CNN) — Log-Mel Spectrogram
# ──────────────────────────────────────────────────────────────────────
TARGET_SPEC_FRAMES = 63  # 2 sec @ 16kHz, hop=512 → ~63 frames

def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (128, TARGET_SPEC_FRAMES) Log-Mel Spectrogram."""
    melspec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # Pad or truncate to fixed width
    if log_melspec.shape[1] < TARGET_SPEC_FRAMES:
        pad = TARGET_SPEC_FRAMES - log_melspec.shape[1]
        log_melspec = np.pad(log_melspec, ((0, 0), (0, pad)), mode='constant')
    else:
        log_melspec = log_melspec[:, :TARGET_SPEC_FRAMES]

    return log_melspec


def _process_single_file_cnn(args):
    """Worker function for parallel CNN feature extraction."""
    file_path, label = args
    try:
        audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, duration=2.0, mono=True)
        if audio is None or len(audio) < SAMPLE_RATE * 0.1:
            return None
        spec = extract_mel_spectrogram(audio)
        return (spec, label)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────────────
def collect_file_list(base_dir: Path, split: str = "training", limit: int = None):
    """
    Collects (file_path, label) tuples from the FoR folder structure.
    label: 0 = real, 1 = fake
    """
    split_dir = base_dir / split
    files = []

    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Missing: {class_dir}")
            continue

        class_files = []
        for ext in SUPPORTED_EXTENSIONS:
            class_files.extend(list(class_dir.rglob(f"*{ext}")))

        if limit:
            class_files = class_files[:limit]

        files.extend([(f, label) for f in class_files])
        logger.info(f"  {split}/{class_name}: {len(class_files)} files")

    return files


def parallel_extract(file_list, worker_fn, desc="Extracting", max_workers=None):
    """Extract features in parallel using ProcessPoolExecutor."""
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 6)  # cap at 6 to avoid RAM issues

    features = []
    labels = []
    done = 0
    total = len(file_list)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in file_list}
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(f"  {desc}: {done}/{total} ({done*100//total}%) | "
                           f"{rate:.0f} files/s | ETA: {eta:.0f}s")

            result = future.result()
            if result is not None:
                features.append(result[0])
                labels.append(result[1])

    return np.array(features), np.array(labels)


# ──────────────────────────────────────────────────────────────────────
# SVM Training
# ──────────────────────────────────────────────────────────────────────
def train_svm(limit: int = None):
    """
    Full SVM training pipeline on for-norm dataset.
    Uses the EXACT same feature extraction as inference (extract_features_v2).
    No preprocessing (bandpass/CMN/normalization) — just load+resample, matching predict.py.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: SVM MODEL TRAINING (for-norm dataset)")
    logger.info("=" * 60)

    # 1. Collect files
    logger.info("Collecting training files...")
    train_files = collect_file_list(FOR_NORM_DIR, "training", limit=limit)
    logger.info(f"Total training files: {len(train_files)}")

    logger.info("Collecting testing files...")
    test_files = collect_file_list(FOR_NORM_DIR, "testing", limit=limit // 5 if limit else None)
    logger.info(f"Total testing files: {len(test_files)}")

    # Also grab validation for a bigger eval set
    logger.info("Collecting validation files...")
    val_files = collect_file_list(FOR_NORM_DIR, "validation", limit=limit // 5 if limit else None)

    # 2. Extract features
    logger.info("\nExtracting 26-dim SVM features (training)...")
    t0 = time.time()
    X_train, y_train = parallel_extract(train_files, _process_single_file_svm, "Train features")
    logger.info(f"Training features: {X_train.shape} in {time.time()-t0:.1f}s")

    logger.info("\nExtracting 26-dim SVM features (testing)...")
    X_test, y_test = parallel_extract(test_files, _process_single_file_svm, "Test features")
    logger.info(f"Testing features: {X_test.shape}")

    if len(val_files) > 0:
        logger.info("\nExtracting 26-dim SVM features (validation)...")
        X_val, y_val = parallel_extract(val_files, _process_single_file_svm, "Val features")
        logger.info(f"Validation features: {X_val.shape}")
    else:
        X_val, y_val = np.array([]), np.array([])

    # 3. Scale features
    logger.info("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. GridSearchCV (Optimized for 50k+ audio samples)
    logger.info("\nStarting GridSearchCV (5-Fold CV)...")
    param_grid = [
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': ['scale', 'auto']},
        {'kernel': ['linear'], 'C': [1, 10]}
    ]

    svm = SVC(probability=True, random_state=42, cache_size=1000)
    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    t0 = time.time()
    grid.fit(X_train_s, y_train)
    train_time = time.time() - t0

    best = grid.best_estimator_
    logger.info(f"\nGridSearchCV complete in {train_time:.1f}s")
    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%")

    # 5. Evaluate on test set
    y_pred = best.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("\n" + "=" * 50)
    logger.info("SVM — HOLDOUT TEST SET RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {acc * 100:.2f}%")
    logger.info(f"Precision: {prec * 100:.2f}%")
    logger.info(f"Recall:    {rec * 100:.2f}%")
    logger.info(f"F1 Score:  {f1 * 100:.2f}%")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  [{cm[0][0]:5d}  {cm[0][1]:5d}]  ← Real")
    logger.info(f"  [{cm[1][0]:5d}  {cm[1][1]:5d}]  ← Fake")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Real", "Fake"], zero_division=0))

    # Evaluate on validation if available
    if len(X_val) > 0:
        X_val_s = scaler.transform(X_val)
        y_val_pred = best.predict(X_val_s)
        val_acc = accuracy_score(y_val, y_val_pred)
        logger.info(f"Validation Accuracy: {val_acc * 100:.2f}%")

    # 6. Save artifacts
    model_path = ML_MODELS_DIR / "best_svm_model.pkl"
    scaler_path = ML_MODELS_DIR / "feature_scaler.pkl"
    joblib.dump(best, str(model_path))
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"\nSVM model saved:  {model_path}")
    logger.info(f"Scaler saved:     {scaler_path}")
    logger.info("=" * 50)

    return acc


# ──────────────────────────────────────────────────────────────────────
# CNN Training
# ──────────────────────────────────────────────────────────────────────
def train_cnn(limit: int = None, epochs: int = 30, batch_size: int = 32, lr: float = 0.001):
    """
    Full CNN training pipeline on for-2sec + for-rerec merged dataset.
    Uses Log-Mel Spectrograms.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nPyTorch device: {device}")

    logger.info("=" * 60)
    logger.info("PHASE 2: CNN MODEL TRAINING (for-2sec + for-rerec merged)")
    logger.info("=" * 60)

    # 1. Collect files from both for-2sec and for-rerec
    logger.info("Collecting training files (for-2sec)...")
    train_files = collect_file_list(FOR_2SEC_DIR, "training", limit=limit)
    logger.info(f"for-2sec training files: {len(train_files)}")

    if FOR_REREC_DIR.exists():
        logger.info("Collecting training files (for-rerec)...")
        rerec_train = collect_file_list(FOR_REREC_DIR, "training", limit=limit)
        train_files.extend(rerec_train)
        logger.info(f"for-rerec training files: {len(rerec_train)}")

    logger.info(f"Total merged training files: {len(train_files)}")

    logger.info("Collecting testing files (for-2sec)...")
    test_files = collect_file_list(FOR_2SEC_DIR, "testing", limit=limit // 5 if limit else None)
    if FOR_REREC_DIR.exists():
        logger.info("Collecting testing files (for-rerec)...")
        rerec_test = collect_file_list(FOR_REREC_DIR, "testing", limit=limit // 5 if limit else None)
        test_files.extend(rerec_test)
    logger.info(f"Total merged testing files: {len(test_files)}")

    # 2. Extract spectrograms
    logger.info("\nExtracting Mel-Spectrograms (training)...")
    t0 = time.time()
    X_train, y_train = parallel_extract(train_files, _process_single_file_cnn, "Train specs")
    logger.info(f"Training spectrograms: {X_train.shape} in {time.time()-t0:.1f}s")

    logger.info("\nExtracting Mel-Spectrograms (testing)...")
    X_test, y_test = parallel_extract(test_files, _process_single_file_cnn, "Test specs")
    logger.info(f"Testing spectrograms: {X_test.shape}")

    # Add channel dim for CNN: (B, 1, Mels, Time)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # 3. Build model (use the existing SpectrogramCNN from the codebase)
    sys.path.insert(0, str(BACKEND_DIR / "ml_system"))
    from models.deep_learning import SpectrogramCNN

    model = SpectrogramCNN().to(device)
    logger.info(f"Model architecture:\n{model}")

    # 4. Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # 5. Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0.0
    best_epoch = 0

    logger.info(f"\nStarting CNN training ({epochs} epochs, bs={batch_size}, lr={lr})")
    logger.info("-" * 70)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch [{epoch+1:2d}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_path = ML_MODELS_DIR / "best_pytorch_model.pth"
            torch.save(model.state_dict(), str(save_path))

    logger.info("-" * 70)
    logger.info(f"CNN Training Complete. Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    logger.info(f"Model saved: {ML_MODELS_DIR / 'best_pytorch_model.pth'}")
    logger.info("=" * 50)

    return best_val_acc


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VoiceGuard AI — Complete Model Retraining on FoR Dataset")
    parser.add_argument("--model", type=str, choices=["svm", "cnn", "both"], default="both",
                        help="Which model(s) to train (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples PER CLASS for training (None = use all)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="CNN training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="CNN batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="CNN learning rate (default: 0.001)")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  VoiceGuard AI — Complete Model Retraining Pipeline     ║")
    logger.info("║  Dataset: FoR (Fake-or-Real)                           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Dataset root:    {DATASET_ROOT}")
    logger.info(f"Model target:    {args.model}")
    logger.info(f"Sample limit:    {args.limit or 'ALL'}")
    logger.info(f"Output dir:      {ML_MODELS_DIR}")
    logger.info("")

    # Verify dataset exists
    if not FOR_NORM_DIR.exists():
        logger.error(f"for-norm not found at: {FOR_NORM_DIR}")
        sys.exit(1)
    if not FOR_2SEC_DIR.exists() and args.model in ("cnn", "both"):
        logger.error(f"for-2sec not found at: {FOR_2SEC_DIR}")
        sys.exit(1)
    if not FOR_REREC_DIR.exists() and args.model in ("cnn", "both"):
        logger.warning(f"for-rerec not found at {FOR_REREC_DIR} — CNN will train on for-2sec only")

    t_start = time.time()
    results = {}

    # SVM
    if args.model in ("svm", "both"):
        svm_acc = train_svm(limit=args.limit)
        results["SVM"] = svm_acc

    # CNN
    if args.model in ("cnn", "both"):
        cnn_acc = train_cnn(
            limit=args.limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        results["CNN"] = cnn_acc

    # Summary
    total_time = time.time() - t_start
    logger.info("\n" + "═" * 60)
    logger.info("TRAINING COMPLETE — SUMMARY")
    logger.info("═" * 60)
    for model_name, acc in results.items():
        logger.info(f"  {model_name}: {acc * 100 if acc < 1 else acc:.2f}% accuracy")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Artifacts:  {ML_MODELS_DIR}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
