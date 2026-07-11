

# Arrhythmia-Classification-using-1D-CNN

An optimized deep learning framework designed to automate the classification of cardiac arrhythmias using a high-performance **1D Convolutional Neural Network (1D-CNN)**. Built to analyze discrete morphological variations in electrocardiogram (ECG) heartbeats, the system eliminates the processing overhead of heavy 2D-convolutional models, achieving real-time inference speed optimized for edge deployment.

The project features a **Streamlit** web application for end-to-end user interaction, allowing both direct numerical signal evaluation and an optional **OpenCV-powered paper-to-signal conversion extension**.

---

## Core Features

* **1D-CNN Architecture:** Custom convolutional layers engineered for automated morphological feature extraction (e.g., QRS complex detection, wave-segment lengths).
* **Robust Generalization:** Integrated **Dropout** and **Global Average Pooling** layers to enforce structural simplicity and eliminate overfitting.
* **Deterministic Pipeline:** Strict data reproduction and training splits secured via fixed-seed sampling (`random_state=42`).
* **Interactive Frontend:** Deployed via **Streamlit** featuring real-time diagnostic output, class confidence scoring, and visual feedback.
* **Image Conversion Extension:** Includes an integrated 5-stage **OpenCV image-processing module** to strip background grid noise and transform raw paper-graph uploads into clean 1D mathematical arrays.

---

## Architecture & Tech Stack

### Dependencies & Frameworks

* **Deep Learning Core:** `TensorFlow` / `Keras` (Model building, optimization, and `.h5` deployment).
* **Signal & Data Preprocessing:** `NumPy` & `Pandas` (Vector normalization, scaling, and dataframe operations).
* **Evaluation Core:** `Scikit-learn` (Deterministic train-test splitting, confusion matrices).
* **UI & Extension Core:** `Streamlit` (Web framework) & `OpenCV` (Image processing).

### Pipeline Flow

```
[Input Layer] ──► [1D Convolution] ──► [ReLU Activation] ──► [Pooling] ──► [Softmax Output]
(186-Samples)       (Kernel Sliders)       (Non-linearity)     (Dim. Reduction)    (5 Class Probabilities)

```

1. **Preprocessing:** Raw inputs are processed and dynamically scaled using Min-Max Normalization to a $[0, 1]$ bounding range for gradient stability.
2. **Inference:** The `ecg_model.h5` structural matrix calculates dot products of spatial inputs across optimized layers.
3. **Classification:** A final **Softmax** layer translates deep neural activations into clear confidence vectors across 5 distinct target classes.

---

## Repository Structure

```text
├── LICENSE             # MIT License file
├── README.md           # Documentation homepage
├── app1.py             # Streamlit web application & UI
├── ecg_model.h5        # Stored weights and optimized network architecture
├── mitbih_test.csv     # Test dataset evaluation file
└── hm.zip              # Supplementary project resource archive

```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SynthBlade7/Arrhythmia-Classification-using-1D-CNN.git
cd Arrhythmia-Classification-using-1D-CNN

```

### 2. Launch the Application

```bash
streamlit run app1.py

```

---

## Model Optimization Highlights

* **Optimizer:** Adam Optimizer leveraging adaptive moment estimation for accelerated loss convergence.
* **EarlyStopping:** Implemented training callbacks to strictly track validation loss and cut execution early to protect against model memorization.
* **Imbalance Resolution:** Handled class disparities across training boundaries utilizing random oversampling to yield a robust multi-class dataset.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
