"""
VoiceGuard AI: Production Retraining on The Fake-or-Real Dataset
================================================================
Trains the SVM on the FULL dataset (50,000+ files) from S:\project_train.
Uses multiprocessing to speed up feature extraction.
Completely offline — no internet required.
"""
import os, sys, logging, numpy as np, joblib, random, time
from pathlib import Path
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR / "ml_system"))
sys.path.insert(0, str(BACKEND_DIR))

from ml_system.features.preprocess import preprocessor
from ml_system.config import MODELS_DIR, SAMPLE_RATE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import librosa, warnings
warnings.filterwarnings('ignore', category=UserWarning)

DATASET_ROOT = Path(r"S:\project_train\for-original\for-original\training")

def collect_audio_files(folder: Path):
    """Collects ALL audio file paths from a folder."""
    exts = {'.wav', '.mp3', '.mp4', '.flac'}
    files = [f for f in folder.rglob("*") if f.suffix.lower() in exts]
    random.seed(42)
    random.shuffle(files)
    return files


def process_single_file(args):
    """Worker function for parallel processing."""
    fpath, label, sr = args
    try:
        audio, _ = librosa.load(str(fpath), sr=sr, duration=5, mono=True)
        if audio is None or len(audio) < sr // 2:
            return None
            
        cleaned = preprocessor.apply_bandpass_filter(audio)
        cleaned = preprocessor.remove_silence(cleaned)
        if cleaned is None or len(cleaned) < sr // 4:
            return None
            
        cleaned = preprocessor.apply_cmn(cleaned)
        cleaned = preprocessor.normalize_amplitude(cleaned)
        
        # Instantiate extractor locally for thread safety
        from ml_system.features.extract import AudioFeatureExtractor
        local_extractor = AudioFeatureExtractor()
        fv = local_extractor.extract_features(cleaned)
        
        if fv is not None and not np.any(np.isnan(fv)):
            return (fv, 0 if label == "REAL" else 1)
    except Exception:
        pass
    return None


def extract_features_parallel(file_list, label, sr=SAMPLE_RATE):
    """Extracts features using all CPU cores to handle 50,000+ files."""
    logger.info(f"  [{label}] Extracting features for {len(file_list)} files using all CPU cores...")
    
    # Process in chunks with a progress indicator
    results = []
    chunk_size = max(1000, len(file_list) // 20)
    
    for i in range(0, len(file_list), chunk_size):
        chunk = file_list[i:i + chunk_size]
        chunk_args = [(f, label, sr) for f in chunk]
        
        chunk_results = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(process_single_file)(arg) for arg in chunk_args
        )
        
        results.extend([r for r in chunk_results if r is not None])
        
        progress = min(100, int((i + chunk_size) / len(file_list) * 100))
        logger.info(f"  [{label}] Progress: {progress}% ({len(results)} valid features extracted so far)")
        
    X = [r[0] for r in results]
    y = [r[1] for r in results]
    return X, y


def main():
    print("\n" + "=" * 60)
    print("VoiceGuard AI: MASSIVE FULL-DATASET TRAINING STARTED!")
    print("Dataset: S:\\project_train (50,000+ files)")
    print("You can safely DISCONNECT THE INTERNET now. Everything is local.")
    print("Note: Processing ~50,000 files will take significant time.")
    print("Using all CPU cores to speed it up as much as possible.")
    print("=" * 60 + "\n")
    
    time.sleep(3)

    real_dir = DATASET_ROOT / "real"
    fake_dir = DATASET_ROOT / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        logger.error(f"Missing REAL or FAKE directories in {DATASET_ROOT}")
        sys.exit(1)

    # 1. Collect ALL files
    logger.info("Locating ALL files in dataset...")
    real_files = collect_audio_files(real_dir)
    fake_files = collect_audio_files(fake_dir)
    
    # Balance classes strictly based on the minority class
    min_class_size = min(len(real_files), len(fake_files))
    real_files = real_files[:min_class_size]
    fake_files = fake_files[:min_class_size]
    
    logger.info(f"Collected: {len(real_files)} REAL, {len(fake_files)} FAKE (Perfectly Balanced)")

    # 2. Extract features (Parallelized for massive speedup)
    logger.info("Extracting features from REAL samples...")
    X_real, y_real = extract_features_parallel(real_files, "REAL")

    logger.info("Extracting features from FAKE samples...")
    X_fake, y_fake = extract_features_parallel(fake_files, "FAKE")

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    if len(X) == 0:
        logger.error("FATAL: No features could be extracted. Aborting.")
        sys.exit(1)

    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Total: {len(X)} valid samples, {X.shape[1]} features")
    logger.info(f"Final Class distribution: {dict(zip(unique, counts))}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    logger.info(f"Training set: {X_train_s.shape}, Testing set: {X_test_s.shape}")

    # 5. Fast Linear SVM for massive dataset
    logger.info("Training SVM model on 50,000+ samples...")
    # Linear kernel is much faster for very large datasets and less prone to overfitting
    svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_train_s, y_train)
    
    # 6. Evaluate
    y_pred = svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info("=" * 40)
    logger.info(f"FINAL TEST ACCURACY: {acc*100:.2f}%")
    logger.info(f"Confusion Matrix:\n  [{cm[0][0]:4d} {cm[0][1]:4d}] <- Genuine\n  [{cm[1][0]:4d} {cm[1][1]:4d}] <- Fake")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Genuine","Fake"], zero_division=0))

    # 7. Save permanently
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = str(MODELS_DIR / "best_svm_model.pkl")
    scaler_path = str(MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Scaler saved: {scaler_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE! Model is permanently saved.")
    print("You can now close this terminal.")
    print("=" * 60 + "\n")
    
    # Prevent the window from closing instantly if user isn't looking
    time.sleep(600)

if __name__ == "__main__":
    main()
