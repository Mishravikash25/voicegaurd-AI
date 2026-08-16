import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "DATASET-balanced.csv"

# Load the latest version
print("Loading dataset from Kaggle...")
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "birdy654/deep-voice-deepfake-voice-recognition",
  file_path,
)

print("Dataset loaded successfully!")
print("First 5 records:\n", df.head())
