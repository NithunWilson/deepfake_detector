"""
Data Download and Preparation Script
This script helps you download and organize deepfake datasets
"""
import os
import sys
import json
import zipfile
import requests
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import DATA_DIR

class DataManager:
    """Manages data download and organization"""
    
    def __init__(self):
        self.raw_dir = DATA_DIR / "raw"
        self.processed_dir = DATA_DIR / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
        print("Creating sample dataset...")
        
        # Create sample directories
        sample_real_dir = self.processed_dir / "real"
        sample_fake_dir = self.processed_dir / "fake"
        sample_real_dir.mkdir(exist_ok=True)
        sample_fake_dir.mkdir(exist_ok=True)
        
        # Create metadata
        metadata = []
        
        # Create sample real videos (using webcam or images)
        for i in range(5):
            video_path = sample_real_dir / f"real_sample_{i}.mp4"
            self.create_sample_video(video_path, is_real=True)
            metadata.append({
                "video_path": str(video_path),
                "label": 0,
                "dataset": "sample",
                "split": "train" if i < 3 else "test"
            })
        
        # Create sample fake videos
        for i in range(5):
            video_path = sample_fake_dir / f"fake_sample_{i}.mp4"
            self.create_sample_video(video_path, is_real=False)
            metadata.append({
                "video_path": str(video_path),
                "label": 1,
                "dataset": "sample",
                "split": "train" if i < 3 else "test"
            })
        
        # Save metadata
        df = pd.DataFrame(metadata)
        metadata_path = DATA_DIR / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        
        print(f"Created sample dataset with {len(df)} videos")
        print(f"Real: {len(df[df['label']==0])}, Fake: {len(df[df['label']==1])}")
        print(f"Metadata saved to: {metadata_path}")
        
        return df
    
    def create_sample_video(self, output_path, is_real=True, duration=5, fps=30):
        """Create a sample video using synthetic data"""
        import cv2
        import numpy as np
        
        # Create a simple video with moving shapes
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = 640, 480
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_num in range(duration * fps):
            # Create background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if is_real:
                # Real video: natural-looking patterns
                color = (100, 200, 100)  # Greenish
                center_x = width // 2
                center_y = height // 2
                radius = 50 + int(25 * np.sin(frame_num * 0.1))
                cv2.circle(frame, (center_x, center_y), radius, color, -1)
            else:
                # Fake video: artificial patterns
                color = (200, 100, 200)  # Purple
                x = (frame_num * 5) % width
                y = height // 2 + int(100 * np.sin(frame_num * 0.05))
                cv2.rectangle(frame, (x, y), (x + 100, y + 100), color, -1)
            
            # Add text label
            label = "REAL" if is_real else "FAKE"
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Created sample video: {output_path.name}")
    
    def organize_external_dataset(self, source_dir, dataset_name):
        """
        Organize external dataset into our format
        Args:
            source_dir: Directory containing the external dataset
            dataset_name: Name of the dataset (e.g., 'faceforensics')
        """
        print(f"Organizing {dataset_name} dataset...")
        
        # This is a template - you need to adapt based on your dataset structure
        dataset_dir = self.raw_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Copy or move files
        if Path(source_dir).exists():
            for item in Path(source_dir).iterdir():
                if item.is_file():
                    shutil.copy2(item, dataset_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, dataset_dir / item.name)
        
        print(f"Dataset organized in: {dataset_dir}")
        
        # Create metadata based on dataset structure
        self.create_metadata_from_dataset(dataset_dir, dataset_name)
    
    def create_metadata_from_dataset(self, dataset_dir, dataset_name):
        """Create metadata CSV from organized dataset"""
        metadata = []
        
        # Look for video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        # Check for real/fake subdirectories
        real_dir = dataset_dir / "real"
        fake_dir = dataset_dir / "fake"
        
        if real_dir.exists():
            for video_file in real_dir.glob("*"):
                if video_file.suffix.lower() in video_extensions:
                    metadata.append({
                        "video_path": str(video_file),
                        "label": 0,
                        "dataset": dataset_name,
                        "split": "train"  # Will be split later
                    })
        
        if fake_dir.exists():
            for video_file in fake_dir.glob("*"):
                if video_file.suffix.lower() in video_extensions:
                    metadata.append({
                        "video_path": str(video_file),
                        "label": 1,
                        "dataset": dataset_name,
                        "split": "train"
                    })
        
        # If no real/fake subdirs, try to parse from filenames
        if not metadata:
            for video_file in dataset_dir.rglob("*"):
                if video_file.suffix.lower() in video_extensions:
                    label = 0 if "real" in video_file.name.lower() else 1
                    metadata.append({
                        "video_path": str(video_file),
                        "label": label,
                        "dataset": dataset_name,
                        "split": "train"
                    })
        
        if metadata:
            # Add to existing metadata or create new
            metadata_path = DATA_DIR / "metadata.csv"
            if metadata_path.exists():
                existing_df = pd.read_csv(metadata_path)
                new_df = pd.DataFrame(metadata)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = pd.DataFrame(metadata)
            
            combined_df.to_csv(metadata_path, index=False)
            print(f"Added {len(metadata)} videos to metadata")
            print(f"Total videos: {len(combined_df)}")
    
    def download_dfd_sample(self):
        """Download a sample from Deepfake Detection Challenge"""
        import kaggle
        
        print("Downloading DFDC sample...")
        
        # Create directory
        dfdc_dir = self.raw_dir / "dfdc_sample"
        dfdc_dir.mkdir(exist_ok=True)
        
        try:
            # Download a small sample
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'deepfake-detection-challenge/sample-videos',
                path=str(dfdc_dir),
                unzip=True
            )
            print("DFDC sample downloaded successfully")
            
            # Process the sample
            self.organize_external_dataset(dfdc_dir / "sample_videos", "dfdc_sample")
            
        except Exception as e:
            print(f"Error downloading DFDC sample: {e}")
            print("Please download manually from:")
            print("https://www.kaggle.com/competitions/deepfake-detection-challenge/data")
    
    def validate_dataset(self):
        """Validate the dataset and check for issues"""
        metadata_path = DATA_DIR / "metadata.csv"
        
        if not metadata_path.exists():
            print("No metadata found. Please create dataset first.")
            return False
        
        df = pd.read_csv(metadata_path)
        
        print("Dataset Validation Report:")
        print(f"Total videos: {len(df)}")
        print(f"Real videos: {len(df[df['label']==0])}")
        print(f"Fake videos: {len(df[df['label']==1])}")
        
        # Check if files exist
        missing_files = []
        for idx, row in df.iterrows():
            if not Path(row['video_path']).exists():
                missing_files.append(row['video_path'])
        
        if missing_files:
            print(f"\nWarning: {len(missing_files)} files are missing:")
            for f in missing_files[:5]:  # Show first 5
                print(f"  - {f}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files)-5} more")
        else:
            print("\nAll video files exist ✓")
        
        return len(missing_files) == 0

def main():
    """Main function for data management"""
    manager = DataManager()
    
    print("=" * 60)
    print("Deepfake Dataset Manager")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Create sample dataset (for testing)")
        print("2. Organize external dataset")
        print("3. Download DFDC sample (requires Kaggle API)")
        print("4. Validate dataset")
        print("5. Show dataset statistics")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            manager.create_sample_dataset()
        
        elif choice == "2":
            source_dir = input("Enter source directory path: ").strip()
            dataset_name = input("Enter dataset name: ").strip()
            if source_dir and dataset_name:
                manager.organize_external_dataset(source_dir, dataset_name)
        
        elif choice == "3":
            manager.download_dfd_sample()
        
        elif choice == "4":
            manager.validate_dataset()
        
        elif choice == "5":
            metadata_path = DATA_DIR / "metadata.csv"
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                print(f"\nDataset Statistics:")
                print(f"Total videos: {len(df)}")
                print(f"Real: {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
                print(f"Fake: {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")
                
                if 'dataset' in df.columns:
                    print("\nBy dataset:")
                    for dataset in df['dataset'].unique():
                        subset = df[df['dataset'] == dataset]
                        print(f"  {dataset}: {len(subset)} videos")
                
                if 'split' in df.columns:
                    print("\nBy split:")
                    for split in df['split'].unique():
                        subset = df[df['split'] == split]
                        print(f"  {split}: {len(subset)} videos")
            else:
                print("No metadata found. Create dataset first.")
        
        elif choice == "6":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
