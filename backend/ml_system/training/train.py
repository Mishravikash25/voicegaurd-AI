import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_system.config import RAW_DATA_DIR
from ml_system.training.train_model import prepare_dataset, train_and_tune_model, save_artifacts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceGuard AI GMM/SVM Training Strategy")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw audio dataset folder")
    args = parser.parse_args()

    target_dir = args.data_dir
    if not target_dir:
        # Default to extracted Kaggle audio folder or RAW_DATA_DIR
        kaggle_audio_dir = Path(__file__).resolve().parent.parent.parent / "dataset" / "KAGGLE" / "AUDIO"
        if kaggle_audio_dir.exists():
            target_dir = str(kaggle_audio_dir)
        else:
            target_dir = str(RAW_DATA_DIR)

    logger.info(f"=== VoiceGuard AI Training (Target Directory: {target_dir}) ===")
    
    X, y = prepare_dataset(target_dir)
    
    if X is None or len(X) == 0:
        logger.error(f"Training aborted. No valid audio data found in '{target_dir}'.")
        sys.exit(1)
        
    logger.info(f"Dataset preparation complete. Samples: {len(X)}, Feature dimensions: {X.shape[1]}")
    
    best_model, fitted_scaler = train_and_tune_model(X, y)
    save_artifacts(best_model, fitted_scaler)
    
    logger.info("=== GMM/SVM Training Complete ===")
