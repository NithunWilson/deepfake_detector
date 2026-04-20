"""
Utility functions for the deepfake detection system
"""
import torch
import numpy as np
import cv2
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import MODELS_DIR

def setup_device():
    """Setup PyTorch device (GPU/CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                    train_acc, val_acc, filename="checkpoint.pth"):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = MODELS_DIR / filename
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    return filepath

def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    """Load model checkpoint"""
    filepath = MODELS_DIR / filename
    
    if not filepath.exists():
        print(f"Checkpoint not found: {filepath}")
        return None, 0
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"Epoch: {checkpoint['epoch']}, Train Acc: {checkpoint['train_acc']:.2f}%")
    
    return checkpoint, checkpoint['epoch']

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = MODELS_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return plot_path

def plot_confusion_matrix(y_true, y_pred, class_names=['Real', 'Fake']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Add percentage text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, f'{cm_percent[i, j]:.1%}',
                    ha='center', va='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return cm, plot_path

def print_evaluation_report(y_true, y_pred, y_prob=None):
    """Print comprehensive evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    for i, class_name in enumerate(['Real', 'Fake']):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask) * 100
            print(f"{class_name} Accuracy: {class_acc:.2f}%")
    
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score, roc_curve
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"\nAUC-ROC Score: {auc:.4f}")
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            roc_path = MODELS_DIR / "roc_curve.png"
            plt.savefig(roc_path, dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")
    
    print("="*60)

def video_to_frames(video_path, target_fps=30, max_frames=300):
    """Extract frames from video at target FPS"""
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling rate
    if fps > target_fps:
        sample_rate = int(fps / target_fps)
    else:
        sample_rate = 1
    
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")
    
    return frames

def create_video_from_frames(frames, output_path, fps=30):
    """Create video from frames"""
    import cv2
    
    if len(frames) == 0:
        raise ValueError("No frames to create video")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def check_video_integrity(video_path):
    """Check if video file is valid and readable"""
    import cv2
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret:
            return False, "Cannot read frames from video"
        
        # Check frame dimensions
        if frame is None or frame.size == 0:
            return False, "Empty frame"
        
        cap.release()
        return True, "Video is valid"
    
    except Exception as e:
        return False, f"Error checking video: {e}"

def get_model_summary(model):
    """Print model summary"""
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    print("\nLayer breakdown:")
    print("-"*40)
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:20} {num_params:>12,} parameters")
    
    print("="*60)
    
    return total_params, trainable_params
