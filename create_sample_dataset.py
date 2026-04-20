"""
Create a proper sample dataset with enough videos
"""
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import os

# Create directories
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create processed directories
real_dir = DATA_DIR / "processed" / "real"
fake_dir = DATA_DIR / "processed" / "fake"
real_dir.mkdir(parents=True, exist_ok=True)
fake_dir.mkdir(parents=True, exist_ok=True)

def create_sample_video(output_path, is_real=True, duration=3, fps=10):
    """Create a sample video with moving shapes"""
    width, height = 320, 240  # Small size for quick processing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_num in range(duration * fps):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if is_real:
            # Real video: smooth gradient movement
            color = (100, 200, 100)  # Greenish
            center_x = width // 2 + int(50 * np.sin(frame_num * 0.1))
            center_y = height // 2 + int(30 * np.cos(frame_num * 0.08))
            radius = 30 + int(15 * np.sin(frame_num * 0.05))
            cv2.circle(frame, (center_x, center_y), radius, color, -1)
            
            # Add smaller circles
            for i in range(3):
                x = int(width * 0.2 * (i + 1))
                y = int(height * 0.8 * np.sin(frame_num * 0.02 + i))
                cv2.circle(frame, (x, y), 10, (200, 100, 50), -1)
        else:
            # Fake video: artificial, jerky patterns
            color = (200, 100, 200)  # Purple
            x = (frame_num * 8) % width
            y = height // 2 + int(80 * np.sin(frame_num * 0.15))
            size = 40 + int(20 * np.sin(frame_num * 0.3))
            cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
            
            # Add sharp patterns
            if frame_num % 10 == 0:
                cv2.line(frame, (0, 0), (width, height), (255, 255, 0), 3)
        
        # Add text label
        label = "REAL" if is_real else "FAKE"
        color = (0, 255, 0) if is_real else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2, cv2.LINE_AA)
        
        out.write(frame)
    
    out.release()
    print(f"Created: {output_path.name}")

def main():
    """Create a balanced dataset"""
    print("Creating balanced sample dataset...")
    
    metadata = []
    
    # Create 20 real videos
    print("\nCreating REAL videos...")
    for i in range(20):
        video_path = real_dir / f"real_sample_{i:03d}.mp4"
        create_sample_video(video_path, is_real=True)
        
        # Assign to train/val/test splits (70/15/15)
        if i < 14:  # 70% train
            split = "train"
        elif i < 17:  # 15% val
            split = "val"
        else:  # 15% test
            split = "test"
            
        metadata.append({
            "video_path": str(video_path),
            "label": 0,  # 0 = real
            "dataset": "sample",
            "split": split
        })
    
    # Create 20 fake videos
    print("\nCreating FAKE videos...")
    for i in range(20):
        video_path = fake_dir / f"fake_sample_{i:03d}.mp4"
        create_sample_video(video_path, is_real=False)
        
        # Assign to train/val/test splits (70/15/15)
        if i < 14:  # 70% train
            split = "train"
        elif i < 17:  # 15% val
            split = "val"
        else:  # 15% test
            split = "test"
            
        metadata.append({
            "video_path": str(video_path),
            "label": 1,  # 1 = fake
            "dataset": "sample",
            "split": split
        })
    
    # Save metadata
    df = pd.DataFrame(metadata)
    metadata_path = DATA_DIR / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET CREATED SUCCESSFULLY!")
    print("="*50)
    print(f"Total videos: {len(df)}")
    print(f"Real videos: {len(df[df['label']==0])}")
    print(f"Fake videos: {len(df[df['label']==1])}")
    print("\nSplit distribution:")
    print(f"Train: {len(df[df['split']=='train'])} videos")
    print(f"Validation: {len(df[df['split']=='val'])} videos")
    print(f"Test: {len(df[df['split']=='test'])} videos")
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Verify all files exist
    missing = []
    for path in df['video_path']:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print(f"\nWarning: {len(missing)} files are missing")
    else:
        print("\nAll video files exist ✓")
    
    return df

if __name__ == "__main__":
    main()
