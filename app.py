from flask import Flask, render_template, request, session, jsonify
import torch
import os
import numpy as np
import cv2
import time
import json
import traceback
from torch import nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
from werkzeug.utils import secure_filename
from PIL import Image as pImage

# ✅ Use SAME detector as training
from preprocessing.face_detector import FaceDetector

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sm = nn.Softmax(dim=1)

# ✅ Initialize face detector (same as training)
face_detector = FaceDetector(method='haar')

model = None
model_info = {'loaded': False}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ SAME MODEL AS TRAINING
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048):
        super(Model, self).__init__()
        backbone = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers)
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
        return fmap, self.linear1(x_lstm[:, -1, :])


# ✅ DATASET MATCHING TRAINING
class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None):
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.transform = transform

    def __getitem__(self, idx):
        frames = []
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        sample_rate = max(1, int(fps / 10))

        frame_count = 0

        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                try:
                    faces = face_detector.detect_faces(frame, max_faces=1)

                    if faces:
                        face = face_detector.extract_face(
                            frame,
                            faces[0],
                            target_size=(im_size, im_size)
                        )
                        if face is not None:
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            frames.append(self.transform(face))
                        else:
                            fallback = cv2.resize(frame, (im_size, im_size))
                            fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                            frames.append(self.transform(fallback))
                    else:
                        fallback = cv2.resize(frame, (im_size, im_size))
                        fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                        frames.append(self.transform(fallback))
                except:
                    fallback = cv2.resize(frame, (im_size, im_size))
                    fallback = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
                    frames.append(self.transform(fallback))

            frame_count += 1

        cap.release()

        if len(frames) == 0:
            blank = torch.zeros(3, im_size, im_size)
            frames = [blank] * self.sequence_length
        elif len(frames) < self.sequence_length:
            last = frames[-1]
            while len(frames) < self.sequence_length:
                frames.append(last.clone())

        return torch.stack(frames).unsqueeze(0)

    def __len__(self):
        return 1


def load_model():
    global model, model_info
    try:
        model = Model(num_classes=2).to(device)

        model_files = [f for f in os.listdir("models") if f.endswith(".pth")]
        if not model_files:
            print("No model found")
            return

        model_path = os.path.join("models", model_files[0])

        # ✅ SAFE LOAD
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        model.eval()

        model_info = {
            'loaded': True,
            'name': model_files[0]
        }

        print(f"Model loaded: {model_files[0]}")

    except Exception as e:
        print("Error loading model:", e)
        traceback.print_exc()


print("Loading model...")
load_model()


@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file'})

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No filename'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'})

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    session['video'] = path
    return jsonify({'success': True})


@app.route('/process')
def process():
    if 'video' not in session:
        return jsonify({'error': 'No video'})

    video_path = session['video']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = VideoDataset(video_path, transform=transform)

    try:
        with torch.no_grad():
            data = dataset[0].to(device)
            _, logits = model(data)
            probs = sm(logits)

            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100

            # ✅ FIXED LABEL MAPPING
            output = "FAKE" if pred == 1 else "REAL"

        return jsonify({
            'output': output,
            'confidence': round(confidence, 2),
            'model_loaded': model_info['loaded']
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Server running at http://127.0.0.1:5000")
    app.run(debug=False)
