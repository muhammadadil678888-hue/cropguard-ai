"""
train.py
Run this once (locally, with internet access for MobileNetV2 ImageNet
weights) before launching the Streamlit app.

    python generate_sample_data.py   # or drop in your real dataset
    python train.py

Produces:
    saved_models/baseline_cnn.keras
    saved_models/mobilenet.keras
    saved_models/metrics.json   <- used by the Evaluation Module in app.py
"""

import os
from model import (
    load_data, preprocess_data, build_baseline_cnn, build_mobilenet_model,
    train_model, evaluate_model, save_metrics,
)

OUT_DIR = "saved_models"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    train_ds, test_ds = load_data("data")
    train_ds_p = preprocess_data(train_ds, augment=True)
    test_ds_p = preprocess_data(test_ds, augment=False)

    results = {}

    print("\n=== Training baseline CNN (from scratch) ===")
    baseline = build_baseline_cnn()
    train_model(baseline, train_ds_p, epochs=10)
    baseline.save(os.path.join(OUT_DIR, "baseline_cnn.keras"))
    results["baseline_cnn"] = evaluate_model(baseline, test_ds_p)
    print("Baseline metrics:", results["baseline_cnn"])

    print("\n=== Training MobileNetV2 transfer-learning model ===")
    mobilenet = build_mobilenet_model()
    train_model(mobilenet, train_ds_p, epochs=8)
    mobilenet.save(os.path.join(OUT_DIR, "mobilenet.keras"))
    results["mobilenet"] = evaluate_model(mobilenet, test_ds_p)
    print("MobileNetV2 metrics:", results["mobilenet"])

    save_metrics(results, os.path.join(OUT_DIR, "metrics.json"))
    print(f"\nSaved models + metrics to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
