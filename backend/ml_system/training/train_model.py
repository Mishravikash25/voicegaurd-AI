import os
import sys
import logging
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

# Ensure we can import from the ml_system root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_system.config import RAW_DATA_DIR, MODELS_DIR, SAMPLE_RATE, DURATION, TEST_SIZE, RANDOM_STATE
from ml_system.utils.data_loader import AudioDataLoader
from ml_system.features.preprocess import preprocessor
from ml_system.features.extract import extractor

try:
    from ml_system.training.visualize import generate_all_visuals
except ImportError:
    generate_all_visuals = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_dataset(data_dir: str):
    """
    Loads raw audio, preprocesses it, and extracts feature vectors.
    """
    logger.info(f"Starting dataset preparation from: {data_dir}")
    loader = AudioDataLoader(sample_rate=SAMPLE_RATE, duration=DURATION)
    
    # Load raw audio arrays and numeric labels
    raw_audio_list, labels = loader.load_dataset(data_dir)
    
    if not raw_audio_list:
        logger.error("No data loaded. Cannot proceed with training.")
        return None, None

    X_features = []
    y_valid = []

    logger.info(f"Extracting features for {len(raw_audio_list)} samples...")
    for idx, (audio, label) in enumerate(zip(raw_audio_list, labels)):
        if idx % 50 == 0 and idx > 0:
            logger.info(f"Processed {idx}/{len(raw_audio_list)} samples...")

        # 1. Preprocess (clean, filter, normalize amplitude)
        cleaned_audio = preprocessor.apply_bandpass_filter(audio)
        cleaned_audio = preprocessor.remove_silence(cleaned_audio)
        if cleaned_audio is None or len(cleaned_audio) == 0:
            continue
        cleaned_audio = preprocessor.apply_cmn(cleaned_audio)
        cleaned_audio = preprocessor.normalize_amplitude(cleaned_audio)

        # 2. Extract Features (1D vector, Unscaled)
        # We disable internal scaling in extractor for global scaling later
        extractor.scaler = None # temporarily disable internal scaler to get raw vectors
        
        feature_vector = extractor.extract_features(cleaned_audio)

        if feature_vector is not None:
            X_features.append(feature_vector)
            y_valid.append(label)

    return np.array(X_features), np.array(y_valid)

def train_and_tune_model(X, y):
    """
    Splits data, applies global scaling, performs K-Fold CV & hyperparameter tuning on SVM,
    evaluates the best model, and returns (best_model, scaler).
    """
    logger.info("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y  # Ensure balanced classes in splits
    )
    
    # 1. Global Feature Scaling (Crucial for SVMs)
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set shape: {X_train_scaled.shape}")
    logger.info(f"Testing set shape: {X_test_scaled.shape}")

    # 2. Hyperparameter Tuning with 5-Fold Cross Validation
    logger.info("Initializing GridSearchCV for Hyperparameter Tuning (5-Fold CV)...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01] # Important for RBF/Poly
    }
    
    svm = SVC(probability=True, random_state=RANDOM_STATE)
    
    # GridSearchCV tests all combinations and uses 5-Fold CV to find the best
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5, 
        scoring='accuracy',
        n_jobs=-1, # Use all available cores
        verbose=1
    )
    
    logger.info("Starting training and tuning process...")
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    logger.info("="*40)
    logger.info("Hyperparameter Tuning Complete")
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")
    logger.info("="*40)

    # 3. Final Evaluation on Holdout Test Set
    logger.info("\nEvaluating the BEST model on unseen test set...")
    y_pred = best_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info("\n" + "="*40)
    logger.info("FINAL MODEL EVALUATION RESULTS")
    logger.info("="*40)
    logger.info(f"Accuracy:  {accuracy * 100:.2f}%")
    logger.info(f"Precision: {precision * 100:.2f}%")
    logger.info(f"Recall:    {recall * 100:.2f}%\n")
    logger.info("Confusion Matrix:")
    logger.info(f"[{conf_matrix[0][0]:3d}  {conf_matrix[0][1]:3d}]  <-- Genuine")
    logger.info(f"[{conf_matrix[1][0]:3d}  {conf_matrix[1][1]:3d}]  <-- Fake\n")
    logger.info("Detailed Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Genuine", "Fake"], zero_division=0))
    logger.info("="*40 + "\n")

    return best_model, scaler

def save_artifacts(model, scaler, model_name="best_svm_model.pkl", scaler_name="feature_scaler.pkl"):
    """
    Saves the trained optimal model and the fitted scaler to the models directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, model_name)
    scaler_path = os.path.join(MODELS_DIR, scaler_name)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Optimized model saved to: {model_path}")
    logger.info(f"Fitted scaler saved to: {scaler_path}")

if __name__ == "__main__":
    logger.info("=== VoiceGuard AI Advanced Training Pipeline ===")
    
    # 1. Prepare Data
    X, y = prepare_dataset(str(RAW_DATA_DIR))
    
    if X is None or len(X) == 0:
        logger.error("Training aborted due to empty dataset. Please add audio files to data/raw/genuine and data/raw/fake.")
        sys.exit(1)
        
    logger.info(f"Dataset preparation complete. Total samples: {len(X)}, Feature dimensions: {X.shape[1]}")
    
    # 2. Train, Tune, Evaluated
    best_model, fitted_scaler = train_and_tune_model(X, y)
    
    # 3. Save Both Artifacts
    save_artifacts(best_model, fitted_scaler)
    
    # 4. Generate Visualizations
    logger.info("Generating Model Performance Visuals...")
    
    # Generate generic feature names since features are combined dynamically
    feature_count = X.shape[1]
    generic_names = [f"Acoustic Feature {i}" for i in range(feature_count)]
    
    # We need to recreate the exact test set split to evaluate properly without data leakage
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_test_scaled = fitted_scaler.transform(X_test)
    
    generate_all_visuals(best_model, X_test_scaled, y_test, feature_names=generic_names)
    
    logger.info("=== Advanced Training Pipeline Complete ===")
