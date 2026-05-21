![Banner](./screenshots/banner.png)

# ✍️ Handwritten Digit Detector

A deep learning web app that detects **handwritten digits (0–9)** from uploaded images using a **CNN trained on the MNIST dataset**, with real-time predictions and confidence scores.

---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-6.x-purple?style=for-the-badge&logo=gradio)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow?style=for-the-badge)

---

### 📑 Table of Contents

* [Live Demo](#-live-demo)
* [Screenshots](#-screenshots)
* [Tech Stack](#-tech-stack)
* [Model Architecture & Performance](#-model-architecture--performance)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [How to Run Locally](#-how-to-run-locally)
* [Contact](#-contact)

---

### 🌐 Live Demo

👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mmuzammilirshad/Hand_Written_Digits_Detector)

---

### 📸 Screenshots

| Upload & Predict | Confidence Results |
|---|---|
| ![Upload](./screenshots/upload.png) | ![Result](./screenshots/result.png) |

---

### 🚀 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

### 🧠 Model Architecture & Performance

The model is a **Convolutional Neural Network (CNN)** built with Keras/TensorFlow.

| Layer | Details |
|---|---|
| Conv2D × 4 | filters=10, kernel=3×3, activation=ReLU |
| MaxPooling2D × 2 | pool_size=2×2 |
| Flatten | — |
| Dense (output) | 10 units, activation=Softmax |

| Metric | Value |
|---|---|
| Dataset | MNIST |
| Training Samples | 60,000 |
| Test Samples | 10,000 |
| Epochs | 5 |
| Test Accuracy | ~98% |

---

### 📊 Dataset

**MNIST** (Modified National Institute of Standards and Technology) is the benchmark dataset for handwritten digit recognition.

| Property | Details |
|---|---|
| Total Images | 70,000 grayscale images |
| Training Set | 60,000 images |
| Test Set | 10,000 images |
| Image Size | 28 × 28 pixels |
| Classes | 10 (digits 0–9) |
| Format | White digit on black background |

> The dataset is loaded directly via `tensorflow.keras.datasets.mnist` — no manual download required.

---

### 📂 Project Structure

```
Hand_Written_Digits_Detector/
│
├── app.py                        # Gradio UI + inference logic
├── requirements.txt              # Pinned dependencies
├── README.md                     # Hugging Face Space config
│
├── saved_model/                  # TensorFlow SavedModel
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   └── variables/
│       ├── variables.index
│       └── variables.data-00000-of-00001
│
├── samples/                      # MNIST-style example images
│   ├── 0.png
│   ├── 1.png
│   ├── 3.png
│   ├── 5.png
│   ├── 7.png
│   └── 9.png
│
├── screenshots/                  # README screenshots
│   ├── banner.png
│   ├── upload.png
│   └── result.png
│
└── Hand_Written_Digits_Dataset.ipynb   # Training notebook
```

---

### ⚙️ How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/Muzammil-ML-Projects/Hand_Written_Digits_Detector.git
   cd Hand_Written_Digits_Detector
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   python app.py
   ```

4. Open in browser
   ```
   http://localhost:7860
   ```

> **Note:** Make sure the `saved_model/` folder is present in the root directory before running.

---

### 📬 Contact

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:cornerofcodes00@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-muzammil-irshad-05b863333)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Muzammil-ML-Projects)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

---

<p align="center">Made with ❤️ by <strong>Muhammad Muzammil Irshad</strong></p>
