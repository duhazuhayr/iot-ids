"""
Adaptive IoT IDS — Master Launcher
====================================
Run this file to execute the full pipeline in order.
Usage:  python run.py
"""

import sys
import os

# ✅ Fix: Make sure Python can find the 'models' and 'utils' packages
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # Always run from project root

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def banner(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 1/6 — Generating IoT Dataset")
from scripts.generate_iot_data import generate_iot_dataset
generate_iot_dataset()

banner("STEP 2/6 — Preprocessing Data")
from utils.preprocess import preprocess_data
preprocess_data()

banner("STEP 3/6 — Training Teacher Model")
from scripts.train_teacher import train_teacher
train_teacher()

banner("STEP 4/6 — Knowledge Distillation (Student Model)")
from scripts.distill_student import train_student_distillation
train_student_distillation()

banner("STEP 5/6 — Training Autoencoder (Anomaly Detector)")
from scripts.train_autoencoder import train_autoencoder
train_autoencoder()

banner("STEP 6/6 — Compressing Student Model")
from scripts.compress_model import compress_model
compress_model()

banner("FINAL — Evaluating Hybrid System")
from scripts.evaluate_hybrid import evaluate_hybrid_system
evaluate_hybrid_system()

print("\n" + "="*60)
print("  [DONE] ALL DONE! Results saved to outputs/")
print("  [NEXT] Now run:  streamlit run app/dashboard.py")
print("="*60 + "\n")
