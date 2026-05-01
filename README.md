# Adaptive Lightweight IoT Intrusion Detection System (IDS)

This project implements a research-level hybrid IDS for IoT environments, combining deep learning classification with unsupervised anomaly detection to identify both known and zero-day attacks.

## Core Features
- **Knowledge Distillation**: A lightweight Student MLP model is trained using a deep Teacher ResNet-style MLP to achieve high accuracy with minimal computation.
- **Model Compression**: Pruning (30%) and Quantization (8-bit) are applied to the student model for deployment on edge devices.
- **Hybrid Detection Logic**: 
    - High-confidence samples are processed by the Student Classifier.
    - Low-confidence samples are routed to a Deep Autoencoder for Anomaly Detection (Zero-Day/Unseen attack identifies).
- **Research focus**: Optimized for the TON_IoT/BoT-IoT simulated data patterns.

## Project Structure
- `data/`: Contains the generated dataset and preprocessing artifacts.
- `models/`: Model definitions and saved weights.
- `scripts/`:
    - `generate_iot_data.py`: Synthesizes realistic IoT traffic.
    - `train_teacher.py`: Trains the high-complex model.
    - `distill_student.py`: Transfers knowledge to the lightweight model.
    - `train_autoencoder.py`: Learns 'Normal' traffic patterns.
    - `compress_model.py`: Optimizes the student model.
    - `evaluate_hybrid.py`: Benchmarks the full system.
- `app/`: Dashboard for real-time visualization (Streamlit).

## Methodology
1. **Teacher Model**: Trained on "Seen" attacks (DoS, DDoS, Injection).
2. **Distillation**: The student model learns from the soft probabilities of the teacher, helping it generalize better even with fewer parameters.
3. **Anomaly detection**: An autoencoder is trained only on "Normal" traffic. Any input with a reconstruction error above a percentile threshold is flagged as an anomaly.
4. **Zero-Day Logic**: If the classifier is unsure about a sample, the autoencoder checks if it looks "normal". If not, it's flagged as a previously unseen attack (e.g., Scanning or Backdoor).

## Setup
```bash
pip install -r requirements.txt
python scripts/generate_iot_data.py
python utils/preprocess.py
python scripts/train_teacher.py
# ... run remaining scripts in order
```

## Results
Detailed metrics can be found in `outputs/summary.txt` and `outputs/confusion_matrix.png` after running the evaluation.
- **Model Size Reduction**: ~45% via pruning and quantization.
- **Zero-Day Detection**: Improved via the confidence-based hybrid routing.
