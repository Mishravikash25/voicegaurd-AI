import os
import sys
import shutil
import logging
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add ml_system directory to sys.path
ml_sys_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_system")
if ml_sys_dir not in sys.path:
    sys.path.insert(0, ml_sys_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_system.training.train_model import prepare_dataset, train_and_tune_model, save_artifacts

def download_in_the_wild_dataset(sample_limit_per_class: int = 250):
    """
    Downloads meta.csv and a balanced subset of 'bona-fide' (REAL) and 'spoof' (FAKE)
    audio files directly from Kaggle using the Kaggle API.
    """
    project_root = Path(__file__).resolve().parent
    base_target = project_root / "dataset" / "KAGGLE" / "AUDIO"
    real_dir = base_target / "REAL"
    fake_dir = base_target / "FAKE"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    meta_path = base_target / "meta.csv"

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    if not meta_path.exists():
        logger.info("Downloading meta.csv from Kaggle dataset 'abdallamohamed312/in-the-wild-audio-deepfake'...")
        api.dataset_download_file('abdallamohamed312/in-the-wild-audio-deepfake', file_name='meta.csv', path=str(base_target))

    if not meta_path.exists():
        logger.error("Failed to acquire meta.csv.")
        return

    logger.info("Reading meta.csv dataset index...")
    df = pd.read_csv(meta_path)

    real_df = df[df['label'] == 'bona-fide'].head(sample_limit_per_class)
    fake_df = df[df['label'] == 'spoof'].head(sample_limit_per_class)

    logger.info(f"Downloading {len(real_df)} REAL (bona-fide) and {len(fake_df)} FAKE (spoof) audio samples...")

    # Download Real samples
    downloaded_real = 0
    for _, row in real_df.iterrows():
        fname = row['file']
        dest_file = real_dir / f"wild_{fname}"
        if dest_file.exists():
            downloaded_real += 1
            continue
        try:
            # Try fetching from release_in_the_wild/real or release_in_the_wild
            remote_path = f"release_in_the_wild/real/{fname}"
            api.dataset_download_file('abdallamohamed312/in-the-wild-audio-deepfake', file_name=remote_path, path=str(real_dir))
            # Move downloaded file to target if needed
            downloaded_name = real_dir / fname
            if downloaded_name.exists():
                downloaded_name.rename(dest_file)
            downloaded_real += 1
        except Exception as e:
            try:
                # Fallback to direct path
                remote_path = f"release_in_the_wild/{fname}"
                api.dataset_download_file('abdallamohamed312/in-the-wild-audio-deepfake', file_name=remote_path, path=str(real_dir))
                downloaded_name = real_dir / fname
                if downloaded_name.exists():
                    downloaded_name.rename(dest_file)
                downloaded_real += 1
            except Exception as e2:
                logger.warning(f"Could not download {fname}: {e2}")

    # Download Fake samples
    downloaded_fake = 0
    for _, row in fake_df.iterrows():
        fname = row['file']
        dest_file = fake_dir / f"wild_{fname}"
        if dest_file.exists():
            downloaded_fake += 1
            continue
        try:
            remote_path = f"release_in_the_wild/fake/{fname}"
            api.dataset_download_file('abdallamohamed312/in-the-wild-audio-deepfake', file_name=remote_path, path=str(fake_dir))
            downloaded_name = fake_dir / fname
            if downloaded_name.exists():
                downloaded_name.rename(dest_file)
            downloaded_fake += 1
        except Exception as e:
            logger.warning(f"Could not download fake/{fname}: {e}")

    logger.info(f"Kaggle 'In-The-Wild' Download Complete: {downloaded_real} REAL, {downloaded_fake} FAKE samples ready.")

def main():
    logger.info("=== VoiceGuard AI: In-The-Wild Audio Deepfake Dataset Downloader & Trainer ===")
    
    # 1. Download balanced subset from Kaggle
    download_in_the_wild_dataset(sample_limit_per_class=100)

    target_dir = Path(__file__).resolve().parent / "dataset" / "KAGGLE" / "AUDIO"

    logger.info(f"Extracting features & preparing training dataset from: {target_dir}")
    X, y = prepare_dataset(str(target_dir))

    if X is None or len(X) == 0:
        logger.error("No valid audio files found for training.")
        sys.exit(1)

    logger.info(f"Successfully prepared {len(X)} total audio feature vectors (Dimensions: {X.shape[1]}).")
    best_model, fitted_scaler = train_and_tune_model(X, y)

    logger.info("Saving optimized SVM model and scaler artifacts...")
    save_artifacts(best_model, fitted_scaler)

    logger.info("=== Model Training on Kaggle In-The-Wild Dataset Complete! ===")

if __name__ == "__main__":
    main()
