"""
TRAINING SCRIPT FOR DEEPFAKE DETECTION
Aligned with app.py inference pipeline:
- Uses face detection during training
- Uses same Model architecture as app.py
- Saves compatible .pth files for app.py
"""

import json
import random
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from preprocessing.face_detector import FaceDetector

warnings.filterwarnings("ignore")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("DEEPFAKE DETECTION TRAINING - FACE-ALIGNED PIPELINE")
print("=" * 70)


class Config:
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"

    SEQUENCE_LENGTH = 60
    FACE_SIZE = (112, 112)
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    NUM_WORKERS = 0
    TARGET_FPS = 10
    PATIENCE = 5

    MODELS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()


class RealVideoDataset(Dataset):
    """
    Dataset that extracts faces from videos so training matches app.py inference.
    Labels:
      0 = REAL
      1 = FAKE
    """

    def __init__(self, metadata_df, transform=None, is_train=True):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        self.sequence_length = config.SEQUENCE_LENGTH
        self.face_size = config.FACE_SIZE
        self.target_fps = config.TARGET_FPS

        # Use Haar for easier setup; change to 'dnn' if your OpenCV DNN model files are ready
        self.face_detector = FaceDetector(method="haar")

        print(f"Dataset initialized with {len(self.metadata_df)} videos")
        print(f"Real: {len(self.metadata_df[self.metadata_df['label'] == 0])}")
        print(f"Fake: {len(self.metadata_df[self.metadata_df['label'] == 1])}")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        video_path = self.metadata_df.iloc[idx]["video_path"]
        label = int(self.metadata_df.iloc[idx]["label"])

        try:
            frames = self.extract_faces_from_video(video_path)

            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            else:
                frames = [self.default_transform(frame) for frame in frames]

            video_tensor = torch.stack(frames)  # (seq_len, 3, H, W)
            return video_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            dummy_tensor = torch.zeros(self.sequence_length, 3, *self.face_size)
            return dummy_tensor, torch.tensor(label, dtype=torch.long)

    def extract_faces_from_video(self, video_path):
        """
        Extract face crops from a video. If no face is found in a frame,
        fall back to resized full frame. Pads sequence if too short.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30

        sample_rate = max(1, int(round(fps / self.target_fps))) if fps > self.target_fps else 1

        frames = []
        frame_count = 0

        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                try:
                    faces = self.face_detector.detect_faces(frame, max_faces=1)

                    if faces:
                        face = self.face_detector.extract_face(
                            frame,
                            faces[0],
                            target_size=self.face_size
                        )
                        if face is not None and face.size > 0:
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            frames.append(face)
                        else:
                            fallback = cv2.resize(frame, self.face_size)
                            fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                            frames.append(fallback)
                    else:
                        fallback = cv2.resize(frame, self.face_size)
                        fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                        frames.append(fallback)

                except Exception:
                    fallback = cv2.resize(frame, self.face_size)
                    fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                    frames.append(fallback)

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            blank = np.zeros((self.face_size[1], self.face_size[0], 3), dtype=np.uint8)
            frames = [blank] * self.sequence_length
        elif len(frames) < self.sequence_length:
            last_frame = frames[-1]
            while len(frames) < self.sequence_length:
                frames.append(last_frame.copy())

        return frames[:self.sequence_length]

    def default_transform(self, frame):
        frame = torch.from_numpy(frame).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        frame = frame.permute(2, 0, 1)
        return frame


class Model(nn.Module):
    """
    Model class kept identical to app.py for compatibility.
    """

    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        backbone = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def load_and_split_data(data_dir):
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    labels = []

    extensions = ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"]

    for ext in extensions:
        for video_file in real_dir.glob(f"**/*.{ext}"):
            video_paths.append(str(video_file))
            labels.append(0)

    for ext in extensions:
        for video_file in fake_dir.glob(f"**/*.{ext}"):
            video_paths.append(str(video_file))
            labels.append(1)

    print(f"\nFound {len(video_paths)} videos:")
    print(f"  Real: {labels.count(0)}")
    print(f"  Fake: {labels.count(1)}")

    if len(video_paths) == 0:
        print("\nNo video files found in data/processed/real and data/processed/fake")
        return None, None, None

    df = pd.DataFrame({
        "video_path": video_paths,
        "label": labels
    })

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    print("\nData split:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")

    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        real_count = len(split_df[split_df["label"] == 0])
        fake_count = len(split_df[split_df["label"] == 1])
        print(f"  {split_name} - Real: {real_count}, Fake: {fake_count}")

    return train_df, val_df, test_df


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        _, logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100.0 * correct / max(total, 1)
        })

    epoch_loss = running_loss / max(len(train_loader), 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            _, logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = logits.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({
                "loss": running_loss / (batch_idx + 1)
            })

    if len(all_labels) == 0:
        return 0.0, 0.0, [], []

    val_acc = 100.0 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    val_loss = running_loss / max(len(val_loader), 1)

    return val_loss, val_acc, all_predictions, all_labels


def evaluate_test(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            _, logits = model(inputs)
            _, predicted = logits.max(1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    return test_labels, test_predictions


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print(f"\nPyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading dataset...")
    train_df, val_df, test_df = load_and_split_data(config.DATA_DIR)
    if train_df is None:
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = RealVideoDataset(train_df, transform=transform, is_train=True)
    val_dataset = RealVideoDataset(val_df, transform=transform, is_train=False)
    test_dataset = RealVideoDataset(test_df, transform=transform, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    print("\nCreating model...")
    model = Model(num_classes=2, lstm_layers=1, bidirectional=False).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    real_count = len(train_df[train_df["label"] == 0])
    fake_count = len(train_df[train_df["label"] == 1])

    if real_count == 0 or fake_count == 0:
        print("One class has zero samples. Cannot train.")
        return

    class_weights = torch.tensor(
        [1.0 / real_count, 1.0 / fake_count],
        dtype=torch.float32,
        device=device
    )
    class_weights = class_weights / class_weights.sum() * 2.0

    print(f"Class weights - Real: {class_weights[0]:.3f}, Fake: {class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if len(val_labels) > 0 and len(val_preds) > 0:
            cm = confusion_matrix(val_labels, val_preds)
            if cm.shape == (2, 2):
                print("Val Confusion Matrix:")
                print(f"  Real predicted as Real: {cm[0, 0]}")
                print(f"  Real predicted as Fake: {cm[0, 1]}")
                print(f"  Fake predicted as Real: {cm[1, 0]}")
                print(f"  Fake predicted as Fake: {cm[1, 1]}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            simple_model_path = config.MODELS_DIR / f"best_model_{int(val_acc)}_{config.SEQUENCE_LENGTH}.pth"
            torch.save(model.state_dict(), simple_model_path)

            full_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "config": {
                    "model": "Model",
                    "sequence_length": config.SEQUENCE_LENGTH,
                    "face_size": config.FACE_SIZE,
                    "num_real": int(real_count),
                    "num_fake": int(fake_count),
                    "pytorch_version": torch.__version__
                }
            }

            torch.save(full_checkpoint, config.MODELS_DIR / "best_model_full.pth")
            print(f"Saved best model with Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{config.PATIENCE} epochs")

            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("-" * 70)

    print(f"\nLoading best model with Val Acc: {best_val_acc:.2f}%")
    best_model_path = config.MODELS_DIR / f"best_model_{int(best_val_acc)}_{config.SEQUENCE_LENGTH}.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_labels, test_predictions = evaluate_test(model, test_loader, device)

    if len(test_labels) == 0:
        print("No test data available.")
        return

    cm = confusion_matrix(test_labels, test_predictions)
    report = classification_report(test_labels, test_predictions, target_names=["Real", "Fake"], zero_division=0)
    test_acc = 100.0 * np.sum(np.array(test_predictions) == np.array(test_labels)) / len(test_labels)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    if cm.shape == (2, 2):
        print(f"Real predicted as Real: {cm[0, 0]}")
        print(f"Real predicted as Fake: {cm[0, 1]}")
        print(f"Fake predicted as Real: {cm[1, 0]}")
        print(f"Fake predicted as Fake: {cm[1, 1]}")

    real_acc = 100.0 * cm[0, 0] / max((cm[0, 0] + cm[0, 1]), 1)
    fake_acc = 100.0 * cm[1, 1] / max((cm[1, 0] + cm[1, 1]), 1)

    print(f"\nReal Video Accuracy: {real_acc:.2f}%")
    print(f"Fake Video Accuracy: {fake_acc:.2f}%")
    print(f"Overall Test Accuracy: {test_acc:.2f}%")

    final_model_path = config.MODELS_DIR / f"deepfake_model_{int(test_acc)}_{config.SEQUENCE_LENGTH}.pth"
    torch.save(model.state_dict(), final_model_path)

    metadata = {
        "test_accuracy": float(test_acc),
        "real_accuracy": float(real_acc),
        "fake_accuracy": float(fake_acc),
        "confusion_matrix": cm.tolist(),
        "train_history": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accs,
            "val_accuracies": val_accs
        },
        "config": {
            "model": "Model",
            "sequence_length": config.SEQUENCE_LENGTH,
            "face_size": config.FACE_SIZE,
            "num_real": int(real_count),
            "num_fake": int(fake_count),
            "training_date": time.strftime("%Y-%m-%d"),
            "pytorch_version": torch.__version__
        }
    }

    metadata_path = config.MODELS_DIR / f"model_metadata_{int(test_acc)}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final model saved to: {final_model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
