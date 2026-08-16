import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add ml_system directory to sys.path
ml_sys_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_system")
if ml_sys_dir not in sys.path:
    sys.path.insert(0, ml_sys_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_system.training.train_model import prepare_dataset, train_and_tune_model, save_artifacts

# Set kagglehub cache directory to S: drive to utilize free disk space
os.environ["KAGGLEHUB_CACHE"] = r"S:\kagglehub_cache"

def download_and_ingest_fake_or_real(sample_limit_per_class: int = 150):
    """
    Downloads 'mohammedabdeldayem/the-fake-or-real-dataset' using kagglehub
    and organizes audio files into REAL and FAKE subdirectories.
    """
    try:
        import kagglehub
    except ImportError:
        logger.info("Installing kagglehub package...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        import kagglehub

    logger.info("Downloading dataset 'mohammedabdeldayem/the-fake-or-real-dataset' via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset"))
    logger.info(f"Dataset downloaded to: {dataset_path}")

    project_root = Path(__file__).resolve().parent
    target_base = project_root / "dataset" / "FAKE_OR_REAL"
    real_dir = target_base / "REAL"
    fake_dir = target_base / "FAKE"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Search for audio files in downloaded directory
    all_audio = list(dataset_path.rglob("*.wav")) + list(dataset_path.rglob("*.mp3"))
    logger.info(f"Found {len(all_audio)} total audio files in kagglehub cache.")

    real_files = []
    fake_files = []

    for f in all_audio:
        try:
            rel_parts = [p.lower() for p in f.relative_to(dataset_path).parts]
            rel_str = "/".join(rel_parts)
            if "for-original" in rel_str or "real" in rel_str or "bona-fide" in rel_str:
                real_files.append(f)
            elif "for-2sec" in rel_str or "for-norm" in rel_str or "for-rerec" in rel_str or "fake" in rel_str or "spoof" in rel_str:
                fake_files.append(f)
        except Exception:
            pass

    logger.info(f"Categorized: {len(real_files)} REAL audio files, {len(fake_files)} FAKE audio files.")

    # Select subset up to sample_limit_per_class
    real_subset = real_files[:sample_limit_per_class]
    fake_subset = fake_files[:sample_limit_per_class]

    for idx, src in enumerate(real_subset):
        dst = real_dir / f"real_{idx}_{src.name}"
        if not dst.exists():
            shutil.copy(src, dst)

    for idx, src in enumerate(fake_subset):
        dst = fake_dir / f"fake_{idx}_{src.name}"
        if not dst.exists():
            shutil.copy(src, dst)

    logger.info(f"Ingested {len(real_subset)} REAL files and {len(fake_subset)} FAKE files to {target_base}")
    return target_base

def main():
    logger.info("=== VoiceGuard AI: 'The-Fake-Or-Real' Dataset Ingestion & Retraining ===")
    
    # 1. Download and Ingest Dataset
    target_dir = download_and_ingest_fake_or_real(sample_limit_per_class=150)

    logger.info(f"Extracting features & preparing training dataset from: {target_dir}")
    X, y = prepare_dataset(str(target_dir))

    if X is None or len(X) == 0:
        logger.error("No valid audio files found for training.")
        sys.exit(1)

    logger.info(f"Successfully prepared {len(X)} total audio feature vectors (Dimensions: {X.shape[1]}).")
    best_model, fitted_scaler = train_and_tune_model(X, y)

    logger.info("Saving optimized SVM model and scaler artifacts...")
    save_artifacts(best_model, fitted_scaler)

    # Clean up heavy raw audio datasets to keep system lightweight
    logger.info("Purging temporary raw audio files to keep repository lightweight...")
    try:
        shutil.rmtree(target_dir, ignore_errors=True)
        cache_dir = Path(r"S:\kagglehub_cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info("Cleanup completed successfully.")
    except Exception as e:
        logger.warning(f"Cleanup non-fatal warning: {e}")

    logger.info("=== Retraining on 'The-Fake-Or-Real' Dataset Complete! ===")

if __name__ == "__main__":
    main()
