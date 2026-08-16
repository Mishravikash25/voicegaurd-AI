import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    """
    A 2D Convolutional Neural Network designed to process Mel-Spectrograms 
    mapped as images (1 channel, frequency bins, time frames).
    Excellent for capturing spatial-temporal acoustic features (formants, pitch tracks).
    """
    def __init__(self, num_classes=2):
        super(SpectrogramCNN, self).__init__()
        
        # Input shape expected: (Batch, Channels=1, Freq_Bins=n_mels, Time_Frames)
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # The exact input size to fc1 depends on the input spectrogram dimensions
        # AdaptiveAvgPool2d forces the output to a fixed size before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Adaptive pooling to handle variable time-lengths
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Return logits (CrossEntropyLoss expects unnormalized logits)
        return x


class AudioSequenceLSTM(nn.Module):
    """
    A Long Short-Term Memory (LSTM) network designed to process sequences 
    of audio features (like MFCC vectors over time frames).
    Excellent for capturing long-term temporal dependencies in speech.
    """
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=2):
        super(AudioSequenceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch_first=True expects input shape: (Batch, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Extract the output of the last time step for classification
        # out shape: (Batch, Seq_Len, Hidden_Size) -> we want (Batch, Hidden_Size)
        out = out[:, -1, :]
        
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    # Quick syntax test
    print("Testing SpectrogramCNN shapes...")
    mock_spectrogram = torch.randn(8, 1, 128, 200) # (Batch, Channel, Mels, TimeFrames)
    cnn = SpectrogramCNN()
    cnn_out = cnn(mock_spectrogram)
    print(f"CNN Output Shape: {cnn_out.shape} (Expected: 8, 2)")
    
    print("\nTesting AudioSequenceLSTM shapes...")
    mock_sequence = torch.randn(8, 200, 40) # (Batch, TimeFrames, MFCC_Features)
    lstm = AudioSequenceLSTM(input_size=40)
    lstm_out = lstm(mock_sequence)
    print(f"LSTM Output Shape: {lstm_out.shape} (Expected: 8, 2)")

    print("PyTorch model definitions validated.")
