# Deepfake Detection System

A production-oriented deepfake detection system that combines computer vision and deep learning to analyze videos and classify them as **REAL** or **FAKE**. The project includes a full pipeline: dataset preparation, model training, and a web-based inference interface.

---

## Overview

This project implements a sequence-based deep learning model that analyzes facial features across video frames. It ensures consistency between training and inference by using identical preprocessing steps, specifically **face extraction and normalization**.

The system is designed to be modular, extensible, and suitable for experimentation as well as deployment.

---

## Features

* Video-based deepfake detection using deep learning
* Face-focused preprocessing for improved model accuracy
* LSTM-based temporal modeling across video frames
* Flask web interface for uploading and analyzing videos
* Dataset generation and management utilities
* Training pipeline with evaluation metrics and visualization
* Configurable preprocessing, training, and inference parameters

---

## Project Structure

```
.
├── app.py                     # Flask web application
├── train.py                   # Model training script
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions
├── preprocessing/
│   ├── face_detector.py       # Face detection module
│   └── video_processor.py     # Video preprocessing pipeline
├── data/
│   ├── processed/
│   │   ├── real/              # Real videos
│   │   └── fake/              # Fake videos
├── models/                    # Trained model weights
├── static/                    # Output images for UI
├── uploads/                   # Uploaded videos
├── create_sample_dataset.py   # Synthetic dataset generator
├── download_data.py           # Dataset management script
└── test_model.py              # Model evaluation script
```

---

## Model Architecture

The model consists of:

* **Backbone**: ResNeXt-50 (feature extraction)
* **Temporal Layer**: LSTM for sequence modeling
* **Classifier**: Fully connected layer for binary classification

### Pipeline

1. Extract frames from video
2. Detect and crop faces
3. Normalize frames
4. Pass sequence into CNN + LSTM model
5. Output classification (REAL / FAKE)

---

## Installation

### Prerequisites

* Python 3.8+
* pip
* Virtual environment (recommended)

### Setup

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Dataset Preparation

### Option 1: Create Sample Dataset

```bash
python create_sample_dataset.py
```

### Option 2: Add Your Own Data

Place videos in:

```
data/processed/real/
data/processed/fake/
```

Supported formats:

* mp4, avi, mov, mkv, webm

---

## Training the Model

```bash
python train.py
```

### Output

* Best model saved in `models/`
* Training metrics and plots generated
* Metadata stored as JSON

---

## Running the Web Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

### Workflow

1. Upload a video
2. System extracts faces and processes frames
3. Model predicts REAL or FAKE
4. Results displayed with confidence score

---

## Key Design Decision

### Training–Inference Consistency

A major improvement in this project is ensuring that:

* Training uses **face-cropped frames**
* Inference uses **face-cropped frames**

This eliminates distribution mismatch and significantly improves prediction reliability.

---

## Evaluation

The model is evaluated using:

* Accuracy
* Confusion matrix
* Classification report
* Per-class performance (REAL vs FAKE)

Run evaluation:

```bash
python test_model.py
```

---

## Configuration

Key parameters can be adjusted in `config.py`:

* Sequence length
* Frame size
* Learning rate
* Batch size
* Dataset split ratios

---

## Security Considerations

* File uploads are validated for type and size
* Secure filename handling is used
* Debug mode is disabled for production use

---

## Limitations

* Performance depends heavily on dataset quality
* No face tracking across frames (future improvement)
* CPU-based inference may be slow for large videos

---

## Future Improvements

* Face tracking across frames
* Real-time inference optimization (GPU)
* Transformer-based architectures
* Improved dataset diversity
* REST API support

---

## Usage Disclaimer

This project is intended for research and educational purposes. It should not be used as a sole source for verifying media authenticity in critical applications.

---

## Author

Nithun Wilson
Cybersecurity Analyst | Penetration Tester

---
