---
title: Handwritten Digit Detector
emoji: 🔢
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🔢 Handwritten Digit Detector

A CNN-based handwritten digit classifier trained on the MNIST dataset.

## Model Architecture
- 4× Conv2D layers (filters=10, kernel=3×3, ReLU)
- 2× MaxPooling2D
- Flatten → Dense(10, softmax)
- Trained for 5 epochs on 60,000 MNIST images

## Preprocessing Pipeline
1. Resize image to 28×28
2. Convert to grayscale
3. Normalize pixel values to [0, 1]
4. Predict with confidence scores for all 10 digits

## Usage
1. Upload a clear image of a handwritten digit (0–9)
2. Click **Predict Digit** or the result appears automatically
3. View the predicted digit and confidence bar chart

## Files
```
├── app.py           # Gradio UI + inference
├── requirements.txt # Pinned dependencies
├── README.md        # This file
└── saved_model/     # TensorFlow SavedModel (upload this folder)
    ├── fingerprint.pb
    ├── saved_model.pb
    └── variables/
        ├── variables.index
        └── variables.data-00000-of-00001
```
