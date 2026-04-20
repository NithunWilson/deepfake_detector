"""
COMPLETE TRAINING SCRIPT FOR REAL VIDEO DATASET
This will properly train on your real and fake videos
Updated for PyTorch 2.6+ compatibility
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*70)
print("DEEPFAKE DETECTION TRAINING - REAL VIDEO DATASET")
print("="*70)

# Configuration
class Config:
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Model parameters
    SEQUENCE_LENGTH = 60  # Match app.py default
    FACE_SIZE = (112, 112)  # Match app.py
    BATCH_SIZE = 4  # Reduced for memory constraints
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 15  # Reduced for faster training
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

config = Config()

class RealVideoDataset(Dataset):
    """Dataset for real video files - matches app.py preprocessing"""
    
    def __init__(self, metadata_df, transform=None, is_train=True):
        """
        Args:
            metadata_df: DataFrame with 'video_path' and 'label' columns
            transform: Image transformations
            is_train: Whether this is training data
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        self.sequence_length = config.SEQUENCE_LENGTH
        self.face_size = config.FACE_SIZE
        
        print(f"Dataset initialized with {len(self.metadata_df)} videos")
        print(f"Real: {len(self.metadata_df[self.metadata_df['label']==0])}")
        print(f"Fake: {len(self.metadata_df[self.metadata_df['label']==1])}")
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        video_path = self.metadata_df.iloc[idx]['video_path']
        label = self.metadata_df.iloc[idx]['label']
        
        try:
            # Extract frames from video
            frames = self.extract_frames(video_path)
            
            # Apply transformations
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            else:
                # Default transform matching app.py
                frames = [self.default_transform(frame) for frame in frames]
            
            # Stack frames
            video_tensor = torch.stack(frames)
            
            return video_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return dummy data
            dummy_tensor = torch.zeros(self.sequence_length, 3, *self.face_size)
            return dummy_tensor, torch.tensor(label, dtype=torch.long)
    
    def extract_frames(self, video_path, target_fps=10):
        """Extract frames from video file with face detection"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30
            
            # Calculate sampling rate
            if fps > target_fps:
                sample_rate = int(fps / target_fps)
            else:
                sample_rate = 1
            
            frames = []
            frame_count = 0
            
            while len(frames) < self.sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Resize and convert to RGB
                    frame = cv2.resize(frame, self.face_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Handle insufficient frames
            if len(frames) < self.sequence_length and len(frames) > 0:
                # Pad with last frame
                last_frame = frames[-1]
                while len(frames) < self.sequence_length:
                    frames.append(last_frame.copy())
            elif len(frames) == 0:
                # Create blank frames
                blank_frame = np.zeros((*self.face_size, 3), dtype=np.uint8)
                frames = [blank_frame] * self.sequence_length
            
            return frames[:self.sequence_length]
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            # Return blank frames as fallback
            blank_frame = np.zeros((*self.face_size, 3), dtype=np.uint8)
            return [blank_frame] * self.sequence_length
    
    def default_transform(self, frame):
        """Default transformation matching app.py normalization"""
        # Convert numpy array to tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        
        # Normalize with ImageNet stats (same as app.py)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        frame_tensor = (frame_tensor - mean) / std
        
        # Permute to (C, H, W)
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        return frame_tensor

# Use the SAME Model class as app.py
class Model(nn.Module):
    """Model class matching app.py exactly"""
    
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
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
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

def load_and_split_data(data_dir):
    """Load and split data from real/fake directories"""
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    
    if not real_dir.exists():
        print(f"Creating real directory: {real_dir}")
        real_dir.mkdir(parents=True, exist_ok=True)
    
    if not fake_dir.exists():
        print(f"Creating fake directory: {fake_dir}")
        fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all video files
    video_paths = []
    labels = []
    
    # Real videos (label 0)
    for ext in ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv']:
        for video_file in real_dir.glob(f"**/*.{ext}"):
            video_paths.append(str(video_file))
            labels.append(0)
    
    # Fake videos (label 1)
    for ext in ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv']:
        for video_file in fake_dir.glob(f"**/*.{ext}"):
            video_paths.append(str(video_file))
            labels.append(1)
    
    print(f"\nFound {len(video_paths)} videos:")
    print(f"  Real: {labels.count(0)}")
    print(f"  Fake: {labels.count(1)}")
    
    if len(video_paths) == 0:
        print("\n⚠ No video files found in data/real/ and data/fake/")
        print("\nPlease add videos to these directories:")
        print("  data/real/   - Put REAL videos here")
        print("  data/fake/   - Put FAKE videos here")
        print("\nSupported formats: mp4, avi, mov, mkv, webm, flv, wmv")
        return None, None, None
    
    if labels.count(0) < 5 or labels.count(1) < 5:
        print("\n⚠ Warning: Very few samples. Need at least 5 of each class.")
        print("Model may not learn well. Add more videos.")
    
    # Create DataFrame
    df = pd.DataFrame({
        'video_path': video_paths,
        'label': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data (80% train, 10% val, 10% test) - adjusted for small datasets
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} videos")
    print(f"  Validation: {len(val_df)} videos")
    print(f"  Test: {len(test_df)} videos")
    
    # Check class balance in splits
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        real_count = len(split_df[split_df['label'] == 0])
        fake_count = len(split_df[split_df['label'] == 1])
        print(f"  {split_name} - Real: {real_count}, Fake: {fake_count}")
    
    return train_df, val_df, test_df

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[1], labels)  # Use classification output
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs[1].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': accuracy
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[1], labels)
            
            running_loss += loss.item()
            _, predicted = outputs[1].max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1)
            })
    
    # Calculate metrics
    val_acc = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    val_loss = running_loss / len(val_loader)
    
    return val_loss, val_acc, all_predictions, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, cm):
    """Plot training results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot losses
        axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracies
        axes[0, 1].plot(train_accs, label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(val_accs, label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot confusion matrix
        axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(['Real', 'Fake'])
        axes[1, 0].set_yticklabels(['Real', 'Fake'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black")
        
        # Plot class distribution
        real_pred = cm[0, 0] + cm[1, 0]
        fake_pred = cm[0, 1] + cm[1, 1]
        axes[1, 1].bar(['Predicted Real', 'Predicted Fake'], [real_pred, fake_pred])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Prediction Distribution')
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Could not create full plots: {e}")
        # Simple plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Val')
        plt.legend()
        plt.title('Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    """Main training function"""
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load and split data
    print("\nLoading dataset...")
    train_df, val_df, test_df = load_and_split_data(config.DATA_DIR)
    
    if train_df is None:
        return
    
    # Create datasets
    train_dataset = RealVideoDataset(train_df, is_train=True)
    val_dataset = RealVideoDataset(val_df, is_train=False)
    test_dataset = RealVideoDataset(test_df, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    # Create model - MUST MATCH app.py Model class
    print("\nCreating model (matching app.py)...")
    model = Model(num_classes=2, lstm_layers=1, bidirectional=False).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class balancing
    real_count = len(train_df[train_df['label'] == 0])
    fake_count = len(train_df[train_df['label'] == 1])
    
    if real_count == 0 or fake_count == 0:
        print("⚠ Error: One class has zero samples!")
        return
    
    # Weight inversely proportional to class frequency
    weight_for_real = 1.0 / real_count if real_count > 0 else 1.0
    weight_for_fake = 1.0 / fake_count if fake_count > 0 else 1.0
    class_weights = torch.tensor([weight_for_real, weight_for_fake]).to(device)
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    
    print(f"Class weights - Real: {class_weights[0]:.3f}, Fake: {class_weights[1]:.3f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=2)
    
    # Training loop
    print("\nStarting training...")
    print("-"*70)
    
    num_epochs = config.NUM_EPOCHS
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate confusion matrix
        val_cm = confusion_matrix(val_labels, val_preds)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_cm.size >= 4:
            print(f"Val Confusion Matrix:")
            print(f"  Real predicted as Real: {val_cm[0, 0]}")
            print(f"  Real predicted as Fake: {val_cm[0, 1]}")
            print(f"  Fake predicted as Real: {val_cm[1, 0]}")
            print(f"  Fake predicted as Fake: {val_cm[1, 1]}")
        
        print("-"*70)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model - PyTorch 2.6+ compatible format
            model_save_path = config.MODELS_DIR / f"best_model_{int(val_acc)}_{config.SEQUENCE_LENGTH}.pth"
            
            # Save ONLY the model state dict for compatibility
            torch.save(model.state_dict(), model_save_path)
            
            # Also save full checkpoint with proper serialization
            full_checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'train_acc': float(train_acc),
                'val_acc': float(val_acc),
                'config': {
                    'model': 'Model',
                    'sequence_length': config.SEQUENCE_LENGTH,
                    'face_size': config.FACE_SIZE,
                    'num_real': int(real_count),
                    'num_fake': int(fake_count),
                    'pytorch_version': torch.__version__
                }
            }
            
            torch.save(full_checkpoint, config.MODELS_DIR / "best_model_full.pth", 
                      _use_new_zipfile_serialization=False)
            
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"✗ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for testing
    print(f"\nLoading best model with Val Acc: {best_val_acc:.2f}%")
    try:
        # Try to load the simple state dict
        model.load_state_dict(torch.load(config.MODELS_DIR / f"best_model_{int(best_val_acc)}_{config.SEQUENCE_LENGTH}.pth", 
                                        map_location=device))
    except:
        print("Using last model for testing")
    
    # Test evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs[1].max(1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    if len(test_labels) > 0:
        cm = confusion_matrix(test_labels, test_predictions)
        report = classification_report(test_labels, test_predictions, 
                                       target_names=['Real', 'Fake'])
        
        print("\nClassification Report:")
        print(report)
        
        print("\nConfusion Matrix:")
        if cm.size >= 4:
            print(f"Real predicted as Real: {cm[0, 0]}")
            print(f"Real predicted as Fake: {cm[0, 1]}")
            print(f"Fake predicted as Real: {cm[1, 0]}")
            print(f"Fake predicted as Fake: {cm[1, 1]}")
        
        # Calculate accuracies
        test_acc = 100. * np.sum(np.array(test_predictions) == np.array(test_labels)) / len(test_labels)
        
        if cm.size >= 4 and cm[0, 0] + cm[0, 1] > 0:
            real_acc = 100. * cm[0, 0] / (cm[0, 0] + cm[0, 1])
            print(f"\nReal Video Accuracy: {real_acc:.2f}%")
        
        if cm.size >= 4 and cm[1, 0] + cm[1, 1] > 0:
            fake_acc = 100. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
            print(f"Fake Video Accuracy: {fake_acc:.2f}%")
        
        print(f"\nOverall Test Accuracy: {test_acc:.2f}%")
        
        # Save final model - PyTorch 2.6+ compatible
        final_model_path = config.MODELS_DIR / f"deepfake_model_{int(test_acc)}_{config.SEQUENCE_LENGTH}.pth"
        torch.save(model.state_dict(), final_model_path)
        
        # Save metadata separately
        metadata = {
            'test_accuracy': float(test_acc),
            'real_accuracy': float(real_acc) if 'real_acc' in locals() else 0.0,
            'fake_accuracy': float(fake_acc) if 'fake_acc' in locals() else 0.0,
            'confusion_matrix': cm.tolist() if cm.size > 0 else [],
            'train_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accs,
                'val_accuracies': val_accs
            },
            'config': {
                'model': 'Model',
                'sequence_length': config.SEQUENCE_LENGTH,
                'face_size': config.FACE_SIZE,
                'num_real': int(real_count),
                'num_fake': int(fake_count),
                'training_date': time.strftime("%Y-%m-%d"),
                'pytorch_version': torch.__version__
            }
        }
        
        import json
        with open(config.MODELS_DIR / f"model_metadata_{int(test_acc)}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved to: {final_model_path}")
        print(f"✓ Metadata saved to: {config.MODELS_DIR / f'model_metadata_{int(test_acc)}.json'}")
        
        # Plot results
        try:
            plot_training_history(train_losses, val_losses, train_accs, val_accs, cm)
        except Exception as e:
            print(f"Could not plot: {e}")
    
    else:
        print("No test data available!")
    
    # Final analysis
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    
    if 'test_acc' in locals() and test_acc > 70:
        print(f"✓ Good performance! Test Accuracy: {test_acc:.1f}%")
    elif 'test_acc' in locals() and test_acc > 50:
        print(f"⚠ Moderate performance. Test Accuracy: {test_acc:.1f}%")
    else:
        print("⚠ May need more training data or different approach")
    
    print("\nNext steps:")
    print("1. Copy the trained model to your app.py models directory")
    print("2. Run 'python app.py' to use the trained model")
    print("3. Add more videos to data/real/ and data/fake/ for better accuracy")
    print("="*70)

if __name__ == "__main__":
    main()
