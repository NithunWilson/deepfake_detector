"""
Face detection module using multiple methods
"""
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, FACE_DETECTION_CONFIG

class FaceDetector:
    """
    Face detector with multiple backends:
    1. DNN (OpenCV) - Fast and accurate
    2. Haar Cascade - Reliable fallback
    3. Dlib - High accuracy (optional)
    """
    
    def __init__(self, method='dnn', min_face_size=None, confidence=None):
        """
        Initialize face detector
        Args:
            method: 'dnn', 'haar', or 'dlib'
            min_face_size: Minimum face size in pixels
            confidence: Minimum confidence threshold
        """
        self.method = method
        self.min_face_size = min_face_size or FACE_DETECTION_CONFIG['min_face_size']
        self.confidence = confidence or FACE_DETECTION_CONFIG['confidence_threshold']
        self.padding_ratio = FACE_DETECTION_CONFIG['padding_ratio']
        
        # Create face detection models directory
        self.models_dir = MODELS_DIR / "face_detection"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self._init_detector()
        
        print(f"Face detector initialized with method: {method}")
    
    def _init_detector(self):
        """Initialize the selected face detector"""
        if self.method == 'dnn':
            self._init_dnn_detector()
        elif self.method == 'haar':
            self._init_haar_detector()
        elif self.method == 'dlib':
            self._init_dlib_detector()
        else:
            raise ValueError(f"Unknown face detection method: {self.method}")
    
    def _init_dnn_detector(self):
        """Initialize DNN face detector"""
        # Download models if not exists
        self._download_dnn_models()
        
        # Load DNN model
        proto_path = self.models_dir / "deploy.prototxt"
        model_path = self.models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not proto_path.exists() or not model_path.exists():
            print("DNN models not found. Falling back to Haar Cascade.")
            self.method = 'haar'
            self._init_haar_detector()
            return
        
        self.dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
        
        # Try to use GPU
        try:
            self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA for DNN face detection")
        except:
            print("Using CPU for DNN face detection")
    
    def _download_dnn_models(self):
        """Download DNN face detection models"""
        models = {
            "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        }
        
        for filename, url in models.items():
            filepath = self.models_dir / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
    
    def _init_haar_detector(self):
        """Initialize Haar Cascade face detector"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not Path(cascade_path).exists():
            # Try to download
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            cascade_path = self.models_dir / "haarcascade_frontalface_default.xml"
            
            if not cascade_path.exists():
                print("Downloading Haar Cascade...")
                try:
                    urllib.request.urlretrieve(cascade_url, cascade_path)
                except:
                    print("Could not download Haar Cascade")
                    raise RuntimeError("Haar Cascade not available")
        
        self.haar_cascade = cv2.CascadeClassifier(str(cascade_path))
    
    def _init_dlib_detector(self):
        """Initialize Dlib face detector"""
        try:
            import dlib
            # Download shape predictor if needed
            predictor_path = self.models_dir / "shape_predictor_68_face_landmarks.dat"
            if not predictor_path.exists():
                print("Dlib shape predictor not found. Using DNN instead.")
                self.method = 'dnn'
                self._init_dnn_detector()
                return
            
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(str(predictor_path))
        except ImportError:
            print("Dlib not installed. Using DNN instead.")
            self.method = 'dnn'
            self._init_dnn_detector()
    
    def detect_faces(self, image, max_faces=1):
        """
        Detect faces in an image
        Args:
            image: Input image (BGR or RGB)
            max_faces: Maximum number of faces to return
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.method == 'dnn':
            return self._detect_faces_dnn(image, max_faces)
        elif self.method == 'haar':
            return self._detect_faces_haar(image, max_faces)
        elif self.method == 'dlib':
            return self._detect_faces_dlib(image, max_faces)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _detect_faces_dnn(self, image, max_faces):
        """Detect faces using DNN"""
        (h, w) = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Pass blob through network
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Check minimum face size
                face_width = x2 - x1
                face_height = y2 - y1
                
                if face_width >= self.min_face_size and face_height >= self.min_face_size:
                    faces.append((x1, y1, x2, y2, confidence))
        
        # Sort by confidence and limit to max_faces
        faces.sort(key=lambda x: x[4], reverse=True)
        faces = faces[:max_faces]
        
        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in faces]
    
    def _detect_faces_haar(self, image, max_faces):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        # Convert format
        detected_faces = []
        for (x, y, w, h) in faces[:max_faces]:
            detected_faces.append((x, y, x + w, y + h))
        
        return detected_faces
    
    def _detect_faces_dlib(self, image, max_faces):
        """Detect faces using Dlib"""
        import dlib
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.dlib_detector(gray, 1)
        
        faces = []
        for detection in detections[:max_faces]:
            x1 = detection.left()
            y1 = detection.top()
            x2 = detection.right()
            y2 = detection.bottom()
            faces.append((x1, y1, x2, y2))
        
        return faces
    
    def extract_face(self, image, face_box=None, target_size=(112, 112)):
        """
        Extract and align face from image
        Args:
            image: Input image
            face_box: Face bounding box (if None, detect first)
            target_size: Output face size
        Returns:
            Cropped and aligned face image
        """
        if face_box is None:
            faces = self.detect_faces(image, max_faces=1)
            if not faces:
                return None
            face_box = faces[0]
        
        x1, y1, x2, y2 = face_box
        
        # Add padding
        padding_w = int((x2 - x1) * self.padding_ratio)
        padding_h = int((y2 - y1) * self.padding_ratio)
        
        x1 = max(0, x1 - padding_w)
        y1 = max(0, y1 - padding_h)
        x2 = min(image.shape[1], x2 + padding_w)
        y2 = min(image.shape[0], y2 + padding_h)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        
        return face
    
    def extract_faces_from_video(self, video_path, max_frames=150, target_fps=30):
        """
        Extract faces from video
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            target_fps: Target frames per second
        Returns:
            List of face images
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default if cannot get fps
        
        # Calculate sampling rate
        if fps > target_fps:
            sample_rate = int(fps / target_fps)
        else:
            sample_rate = 1
        
        faces = []
        frame_count = 0
        
        while len(faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Detect and extract face
                face = self.extract_face(frame)
                if face is not None:
                    faces.append(face)
                else:
                    # If no face detected, skip
                    pass
            
            frame_count += 1
        
        cap.release()
        
        # Handle insufficient faces
        if len(faces) < max_frames and len(faces) > 0:
            # Pad with last face
            last_face = faces[-1]
            while len(faces) < max_frames:
                faces.append(last_face.copy())
        elif len(faces) == 0:
            # No faces found, create blank faces
            blank_face = np.zeros((112, 112, 3), dtype=np.uint8)
            faces = [blank_face] * max_frames
        
        return faces[:max_frames]
    
    def visualize_detection(self, image, face_boxes, output_path=None):
        """
        Visualize face detections on image
        Args:
            image: Input image
            face_boxes: List of face bounding boxes
            output_path: Path to save visualization
        Returns:
            Image with detections drawn
        """
        vis_image = image.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(face_boxes):
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face {i+1}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
        
        return vis_image

# Test the face detector
if __name__ == "__main__":
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test different detectors
    for method in ['dnn', 'haar', 'dlib']:
        try:
            detector = FaceDetector(method=method)
            faces = detector.detect_faces(test_image)
            print(f"{method.upper()} detected {len(faces)} faces")
            
            if faces:
                face = detector.extract_face(test_image, faces[0])
                print(f"Extracted face shape: {face.shape}")
        except Exception as e:
            print(f"{method.upper()} failed: {e}")
