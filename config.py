import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, STATIC_DIR, UPLOADS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "face_size": (112, 112),          # Face crop size
    "sequence_length": 150,           # Number of frames per video
    "target_fps": 30,                 # Target frames per second
    "train_split": 0.7,               # Training data percentage
    "val_split": 0.15,                # Validation data percentage
    "test_split": 0.15,               # Test data percentage
}

# Model configuration
MODEL_CONFIG = {
    "batch_size": 8,                  # Training batch size
    "learning_rate": 0.0001,          # Learning rate
    "weight_decay": 0.001,           # L2 regularization
    "num_epochs": 20,                # Number of epochs
    "patience": 5,                   # Early stopping patience
    "lstm_hidden_size": 512,         # LSTM hidden size
    "lstm_num_layers": 2,            # LSTM layers
}

# Face detection configuration
FACE_DETECTION_CONFIG = {
    "confidence_threshold": 0.7,     # Minimum confidence for face detection
    "min_face_size": 30,            # Minimum face size in pixels
    "padding_ratio": 0.2,           # Padding around detected face
}

# Web application configuration
APP_CONFIG = {
    "port": 5000,                    # Flask port
    "debug": True,                   # Debug mode
    "max_content_length": 500 * 1024 * 1024,  # 500MB max upload
    "allowed_extensions": {"mp4", "avi", "mov", "mkv", "webm"},
}

# Test configuration
TEST_CONFIG = {
    "sample_videos": 10,            # Number of sample videos to create
    "sample_duration": 5,           # Duration of sample videos in seconds
    "sample_fps": 30,               # FPS of sample videos
}
