"""
Video processing module for deepfake detection
"""
import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import DATASET_CONFIG
from .face_detector import FaceDetector

class VideoProcessor:
    """Process videos for deepfake detection"""
    
    def __init__(self, face_detector=None, sequence_length=None, target_fps=None):
        """
        Initialize video processor
        Args:
            face_detector: FaceDetector instance
            sequence_length: Number of frames per sequence
            target_fps: Target frames per second
        """
        self.sequence_length = sequence_length or DATASET_CONFIG['sequence_length']
        self.target_fps = target_fps or DATASET_CONFIG['target_fps']
        
        # Initialize face detector
        if face_detector is None:
            self.face_detector = FaceDetector(method='dnn')
        else:
            self.face_detector = face_detector
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(DATASET_CONFIG['face_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process_video(self, video_path, max_frames=None):
        """
        Process a video for model input
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (None for sequence_length)
        Returns:
            Tensor of shape (sequence_length, 3, H, W)
        """
        max_frames = max_frames or self.sequence_length
        
        # Extract faces from video
        faces = self.face_detector.extract_faces_from_video(
            video_path, 
            max_frames=max_frames,
            target_fps=self.target_fps
        )
        
        # Convert faces to tensors
        face_tensors = []
        for face in faces:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            face_tensor = self.transform(face_rgb)
            face_tensors.append(face_tensor)
        
        # Stack tensors
        if len(face_tensors) == 0:
            # Create blank tensor if no faces found
            face_tensors = [torch.zeros(3, *DATASET_CONFIG['face_size'])] * max_frames
        
        # Ensure we have exactly sequence_length frames
        if len(face_tensors) < self.sequence_length:
            # Pad with last frame
            last_tensor = face_tensors[-1] if face_tensors else torch.zeros(3, *DATASET_CONFIG['face_size'])
            while len(face_tensors) < self.sequence_length:
                face_tensors.append(last_tensor.clone())
        
        face_tensors = face_tensors[:self.sequence_length]
        
        # Stack into single tensor
        video_tensor = torch.stack(face_tensors)  # Shape: (seq_len, 3, H, W)
        
        return video_tensor
    
    def process_video_for_display(self, video_path, max_frames=30):
        """
        Process video for web display (returns images instead of tensors)
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
        Returns:
            List of face images
        """
        # Extract faces from video
        faces = self.face_detector.extract_faces_from_video(
            video_path,
            max_frames=max_frames,
            target_fps=self.target_fps
        )
        
        # Convert BGR to RGB for display
        faces_rgb = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces]
        
        return faces_rgb
    
    def create_preview_frames(self, video_path, num_frames=6):
        """
        Create preview frames from video
        Args:
            video_path: Path to video file
            num_frames: Number of preview frames
        Returns:
            List of preview images
        """
        # Extract evenly spaced frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        
        # Calculate frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        previews = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Detect and extract face
                faces = self.face_detector.detect_faces(frame, max_faces=1)
                if faces:
                    face = self.face_detector.extract_face(frame, faces[0])
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    previews.append(face_rgb)
        
        cap.release()
        
        return previews
    
    def batch_process_videos(self, video_paths, labels=None, batch_size=8):
        """
        Process multiple videos in batches
        Args:
            video_paths: List of video paths
            labels: Optional list of labels
            batch_size: Batch size for processing
        Returns:
            Processed video tensors and labels
        """
        video_tensors = []
        video_labels = []
        
        # Process in batches
        for i in tqdm(range(0, len(video_paths), batch_size), 
                     desc="Processing videos"):
            batch_paths = video_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size] if labels else None
            
            batch_tensors = []
            for video_path in batch_paths:
                try:
                    video_tensor = self.process_video(video_path)
                    batch_tensors.append(video_tensor)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    # Add blank tensor as placeholder
                    blank_tensor = torch.zeros(self.sequence_length, 3, 
                                             *DATASET_CONFIG['face_size'])
                    batch_tensors.append(blank_tensor)
            
            # Stack batch
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors)
                video_tensors.append(batch_tensor)
                
                if batch_labels is not None:
                    batch_label_tensor = torch.tensor(batch_labels, dtype=torch.long)
                    video_labels.append(batch_label_tensor)
        
        # Concatenate all batches
        if video_tensors:
            video_tensors = torch.cat(video_tensors, dim=0)
        
        if video_labels:
            video_labels = torch.cat(video_labels, dim=0)
        
        return video_tensors, video_labels
    
    def save_processed_video(self, video_tensor, output_path, fps=30):
        """
        Save processed video tensor as MP4 file
        Args:
            video_tensor: Video tensor (seq_len, 3, H, W)
            output_path: Output file path
            fps: Frames per second
        """
        from torchvision.utils import save_image
        import tempfile
        import os
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each frame as image
            frame_paths = []
            for i in range(video_tensor.shape[0]):
                frame = video_tensor[i]
                frame_path = Path(temp_dir) / f"frame_{i:04d}.jpg"
                
                # Denormalize
                frame = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                frame = frame + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                frame = torch.clamp(frame, 0, 1)
                
                save_image(frame, frame_path)
                frame_paths.append(str(frame_path))
            
            # Create video from frames
            import imageio
            images = []
            for frame_path in frame_paths:
                images.append(imageio.imread(frame_path))
            
            imageio.mimsave(output_path, images, fps=fps)
        
        print(f"Saved processed video to: {output_path}")

# Test the video processor
if __name__ == "__main__":
    # Create a test video processor
    processor = VideoProcessor()
    
    # Test with a sample video (if exists)
    sample_video = Path("data/processed/real/real_sample_0.mp4")
    
    if sample_video.exists():
        print(f"Processing sample video: {sample_video}")
        
        # Process video
        video_tensor = processor.process_video(sample_video)
        print(f"Video tensor shape: {video_tensor.shape}")
        
        # Create preview
        previews = processor.create_preview_frames(sample_video, num_frames=3)
        print(f"Created {len(previews)} preview frames")
        
        # Save previews
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        for i, preview in enumerate(previews):
            preview_path = output_dir / f"preview_{i}.jpg"
            cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            print(f"Saved preview: {preview_path}")
    else:
        print("Sample video not found. Run download_data.py first.")
