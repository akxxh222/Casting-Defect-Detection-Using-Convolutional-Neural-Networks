# CNN-Based Casting Defect Detection

This project is a deep learning based casting defect inspection system. It uses a Convolutional Neural Network (CNN) to classify grayscale casting surface images as either **OK / Pass** or **Defective**, and includes a Flask web interface for uploading and inspecting images through a simple operator-style dashboard.

## Overview

Industrial casting components can contain surface defects that are difficult and repetitive to inspect manually. This project automates that inspection task by training a binary image classifier on casting images.

The system includes:

- Image preprocessing for grayscale casting images
- CNN training using TensorFlow/Keras
- A saved trained model in `.keras` format
- Single-image prediction script
- Flask web app for image upload and classification
- Browser-based preview, confidence display, and pass/fail status UI

## Features

- Binary classification: `OK` vs `Defective`
- Grayscale image input resized to `150 x 150`
- Data augmentation during training
- Trained CNN model saved as `casting_defect_cnn.keras`
- Flask web application for quick testing
- Drag-and-drop image upload interface
- Confidence score display
- Runtime uploads excluded from Git using `.gitignore`

## Project Structure

```text
CNN-based-defect-detection/
├── app.py                         # Flask web application
├── preprocess.py                  # Dataset preprocessing setup check
├── train_cnn.py                   # CNN model training script
├── predict.py                     # Single-image prediction script
├── check.py                       # Dataset image count checker
├── convert.py                     # Converts .h5 model to .keras format
├── casting_defect_cnn.keras       # Trained CNN model
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── templates/
│   └── index.html                 # Flask HTML template
├── static/
│   ├── app.js                     # Upload preview and drag/drop logic
│   ├── css/
│   │   └── style.css              # Web interface styling
│   └── uploads/
│       └── .gitkeep               # Keeps upload folder in Git
├── train/
│   ├── ok_front/                  # Training images without defects
│   └── def_front/                 # Training images with defects
└── test/
    ├── ok_front/                  # Test images without defects
    └── def_front/                 # Test images with defects
```

## Dataset

The dataset is arranged using the folder structure expected by Keras `flow_from_directory()`.

```text
train/
├── ok_front/
└── def_front/

test/
├── ok_front/
└── def_front/
```

Current dataset count:

| Split | Class | Images |
|---|---:|---:|
| Train | OK | 2875 |
| Train | Defective | 3758 |
| Test | OK | 262 |
| Test | Defective | 453 |

You can verify the counts by running:

```bash
python check.py
```

## Model Architecture

The CNN is defined in `train_cnn.py`.

Architecture summary:

- Input shape: `150 x 150 x 1`
- Convolution block 1: `Conv2D(32)` + `MaxPooling2D`
- Convolution block 2: `Conv2D(64)` + `MaxPooling2D`
- Convolution block 3: `Conv2D(128)` + `MaxPooling2D`
- Flatten layer
- Dense layer with `128` units and ReLU activation
- Dropout layer with `0.5` rate
- Output layer with `1` sigmoid neuron

Training configuration:

- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Metric: `accuracy`
- Batch size: `16`
- Epochs: `15`
- Input color mode: grayscale
- Image size: `150 x 150`

## Requirements

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

Dependencies:

```text
flask
numpy
pillow
tensorflow
```

`pillow` is required because Keras image loading utilities use it internally to read image files such as `.jpg`, `.jpeg`, and `.png`.

## How to Run the Web App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the trained model exists in the project root:

```text
casting_defect_cnn.keras
```

3. Start the Flask app:

```bash
python app.py
```

4. Open the local server in your browser:

```text
http://127.0.0.1:5000
```

5. Upload a casting image and click **Analyze Image**.

The app will:

- Save the uploaded image temporarily inside `static/uploads/`
- Convert the image to grayscale
- Resize it to `150 x 150`
- Normalize pixel values to `0-1`
- Run prediction using `casting_defect_cnn.keras`
- Display the result and confidence score

## How Prediction Works

The model outputs a single sigmoid value between `0` and `1`.

In `app.py` and `predict.py`, the decision rule is:

```python
if prediction >= 0.5:
    result = "PASS - OK"
else:
    result = "DEFECTIVE"
```

Confidence is calculated as:

- OK confidence: `prediction * 100`
- Defective confidence: `(1 - prediction) * 100`

## Training the Model

To train the CNN again from the dataset:

```bash
python train_cnn.py
```

This script will:

1. Load images from `train/` and `test/`
2. Apply augmentation to training images
3. Train the CNN for 15 epochs
4. Evaluate the model on the test set
5. Save the trained model as:

```text
casting_defect_cnn.keras
```

## Single Image Prediction

You can test prediction from the command line using:

```bash
python predict.py
```

By default, `predict.py` uses this image path:

```python
img_path = "test/def_front/cast_def_0_65.jpeg"
```

Change `img_path` inside `predict.py` to test another image.

## Utility Scripts

| File | Purpose |
|---|---|
| `preprocess.py` | Builds Keras image generators and verifies preprocessing setup |
| `train_cnn.py` | Trains and saves the CNN model |
| `predict.py` | Runs prediction on one selected image |
| `check.py` | Prints image counts for train/test class folders |
| `convert.py` | Converts an older `casting_defect_cnn.h5` model into `.keras` format |
| `app.py` | Runs the Flask upload and prediction interface |

## Notes

- Uploaded images are stored in `static/uploads/` during app usage.
- Runtime uploads are ignored by Git through `.gitignore`.
- The trained model file is included in the repository so the app can run without retraining.
- If you retrain the model, `casting_defect_cnn.keras` will be overwritten.

## Future Improvements

- Add validation graphs for accuracy and loss
- Add confusion matrix and classification report
- Add file type validation for uploads
- Add a REST API endpoint for prediction
- Add model versioning for retrained models
