import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure we can import from the ml_system root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, MODELS_DIR, SAMPLE_RATE, DURATION, TEST_SIZE, RANDOM_STATE
from utils.data_loader import AudioDataLoader
from features.preprocess import preprocessor
from features.extract import extractor
from models.deep_learning import SpectrogramCNN, AudioSequenceLSTM

from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using PyTorch device: {device}")


def prepare_dl_dataset(data_dir: str, model_type: str = 'cnn'):
    """
    Loads and prepares data specifically for Deep Learning models.
    'cnn' extracts 2D Mel-Spectrograms.
    'lstm' extracts 2D Sequences of MFCCs over time.
    """
    logger.info(f"Starting DL dataset preparation for {model_type.upper()}...")
    loader = AudioDataLoader(sample_rate=SAMPLE_RATE, duration=DURATION)
    raw_audio_list, labels = loader.load_dataset(data_dir)
    
    if not raw_audio_list:
        logger.error("No data loaded. Cannot proceed.")
        return None, None

    X_features = []
    y_valid = []

    for idx, (audio, label) in enumerate(zip(raw_audio_list, labels)):
        if idx % 50 == 0 and idx > 0:
            logger.info(f"Processed {idx}/{len(raw_audio_list)} samples...")

        cleaned_audio = preprocessor.apply_bandpass_filter(audio)
        cleaned_audio = preprocessor.remove_silence(cleaned_audio)
        if cleaned_audio is None or len(cleaned_audio) == 0:
            continue
        cleaned_audio = preprocessor.apply_cmn(cleaned_audio)
        cleaned_audio = preprocessor.normalize_amplitude(cleaned_audio)

        if model_type == 'cnn':
            features = extractor.extract_mel_spectrogram(cleaned_audio)
        elif model_type == 'lstm':
            features = extractor.extract_sequence_features(cleaned_audio)
        else:
            raise ValueError("Unsupported model_type. Use 'cnn' or 'lstm'")

        if features is not None:
            X_features.append(features)
            y_valid.append(label)

    X = np.array(X_features)
    y = np.array(y_valid)
    
    # CNN expects (Batch, Channels, Height, Width) -> add channel dimension
    if model_type == 'cnn':
        X = np.expand_dims(X, axis=1) # Shape: (B, 1, Mels, Time)
        
    return X, y

def train_dl_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32, lr=0.001):
    """
    Standard PyTorch training loop with validation tracking.
    """
    # Convert numpy arrays to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long) # CrossEntropy expects long (class indices)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("=" * 40)
    logger.info(f"Starting PyTorch Training ({epochs} Epochs)")
    logger.info("=" * 40)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = 100 * correct / total
        
        # Validation Phase
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
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                    f"Train Loss: {train_loss/len(train_loader):.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss/len(test_loader):.4f} | "
                    f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save the best model
            os.makedirs(MODELS_DIR, exist_ok=True)
            save_path = os.path.join(MODELS_DIR, "best_pytorch_model.pth")
            torch.save(model.state_dict(), save_path)
            
    logger.info("=" * 40)
    logger.info(f"Deep Learning Training Complete. Best Val Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {os.path.join(MODELS_DIR, 'best_pytorch_model.pth')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VoiceGuard AI Deep Learning Pipeline")
    parser.add_argument("--model", type=str, choices=['cnn', 'lstm'], default='cnn', 
                        help="Choose deep learning architecture (cnn for spectrograms, lstm for sequences)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw audio dataset folder")
    args = parser.parse_args()
    
    target_dir = args.data_dir
    if not target_dir:
        from pathlib import Path
        kaggle_audio_dir = Path(__file__).resolve().parent.parent.parent / "dataset" / "KAGGLE" / "AUDIO"
        if kaggle_audio_dir.exists():
            target_dir = str(kaggle_audio_dir)
        else:
            target_dir = str(RAW_DATA_DIR)

    logger.info(f"=== VoiceGuard AI PyTorch Upgrade ({args.model.upper()}) ===")
    
    # 1. Prepare Data
    X, y = prepare_dl_dataset(target_dir, model_type=args.model)
    
    if X is None or len(X) == 0:
        logger.error("Training aborted due to empty dataset.")
        sys.exit(1)
        
    logger.info(f"Dataset Shape: {X.shape}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 3. Initialize PyTorch Model
    if args.model == 'cnn':
        # CNN doesn't strictly need input size defined for init in our architecture 
        # because of adaptive pooling.
        dl_model = SpectrogramCNN()
    elif args.model == 'lstm':
        # Get feature dimension (last dimension of X)
        feature_dim = X.shape[-1]
        dl_model = AudioSequenceLSTM(input_size=feature_dim)
        
    # 4. Train
    train_dl_model(dl_model, X_train, y_train, X_test, y_test, epochs=args.epochs)
