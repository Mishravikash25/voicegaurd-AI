import os
import sys
import logging
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Any

# Ensure we can import from the ml_system root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ml_system.config import MODELS_DIR
except ImportError:
    from config import MODELS_DIR

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class to manage the lifecycle of ML models.
    Ensures the model and scaler are loaded into memory only once
    for efficient inference and handles versioned saving.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.scaler = None
            cls._instance.model_path = None
            cls._instance.scaler_path = None
        return cls._instance

    def _get_latest_version(self, prefix: str) -> str:
        """
        Generates a version string based on the current timestamp.
        Format: prefix_vYYYYMMDD_HHMMSS.pkl
        """
        timestamp = datetime.now().strftime("v%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.pkl"

    def save_model(self, model: Any, scaler: Any, base_name: str = "voiceguard_svm") -> Tuple[str, str]:
        """
        Saves the model and scaler using joblib, ensuring version control.
        """
        # Ensure directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Determine paths
        model_filename = self._get_latest_version(f"{base_name}_model")
        scaler_filename = self._get_latest_version(f"{base_name}_scaler")
        
        model_filepath = os.path.join(MODELS_DIR, model_filename)
        scaler_filepath = os.path.join(MODELS_DIR, scaler_filename)
        
        # Save artifacts
        try:
            # We use joblib instead of raw pickle as it is highly optimized 
            # for numpy arrays often found in scikit-learn models
            joblib.dump(model, model_filepath)
            joblib.dump(scaler, scaler_filepath)
            
            # Create a "latest" symlink/copy for easy deployment access
            latest_model_path = os.path.join(MODELS_DIR, f"{base_name}_model_latest.pkl")
            latest_scaler_path = os.path.join(MODELS_DIR, f"{base_name}_scaler_latest.pkl")
            
            joblib.dump(model, latest_model_path)
            joblib.dump(scaler, latest_scaler_path)

            logger.info(f"Successfully saved and versioned model: {model_filename}")
            
            # Update current state if we are saving what we intend to use
            self.model = model
            self.scaler = scaler
            self.model_path = latest_model_path
            self.scaler_path = latest_scaler_path

            return model_filepath, scaler_filepath
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {e}")
            raise

    def load_model(self, model_name: str = "voiceguard_svm_model_latest.pkl", 
                         scaler_name: str = "voiceguard_svm_scaler_latest.pkl") -> Tuple[Any, Any]:
        """
        Loads the model and scaler. If they are already in memory, it returns the cached versions
        to prevent redundant disk I/O during rapid inference.
        """
        # Paths
        target_model_path = os.path.join(MODELS_DIR, model_name)
        target_scaler_path = os.path.join(MODELS_DIR, scaler_name)

        # Check if already loaded
        paths_match = (self.model_path == target_model_path) and (self.scaler_path == target_scaler_path)
        if self.model is not None and self.scaler is not None and paths_match:
            logger.debug("Model and Scaler already in memory. Reusing cached instances.")
            return self.model, self.scaler

        # Proceed to load
        logger.info(f"Loading Model from: {target_model_path}")
        logger.info(f"Loading Scaler from: {target_scaler_path}")
        
        if not os.path.exists(target_model_path) or not os.path.exists(target_scaler_path):
            error_msg = f"Artifacts not found in {MODELS_DIR}. Please train the model first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self.model = joblib.load(target_model_path)
            self.scaler = joblib.load(target_scaler_path)
            
            self.model_path = target_model_path
            self.scaler_path = target_scaler_path
            
            logger.info("Successfully loaded Model and Scaler into memory.")
            return self.model, self.scaler

        except Exception as e:
            logger.error(f"Failed to initialize models from disk: {e}")
            self.model = None
            self.scaler = None
            raise

# Global singleton instance
model_manager = ModelManager()

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 1. Test Singleton Behavior
    manager1 = ModelManager()
    manager2 = ModelManager()
    assert manager1 is manager2, "Singleton instantiation failed."
    logger.info("Singleton validation passed.")
    
    # Normally you'd test loading/saving here, but we don't want to create junk files 
    # unless actual objects are passed.
    logger.info("ModelManager ready.")
