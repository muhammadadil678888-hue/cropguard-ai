"""
model.py
Core Logic Module for CropGuard AI.

Two approaches are implemented so the Evaluation Module has something
real to compare (per lab requirement: "compare at least two settings/
approaches"):

  1. "baseline_cnn"   - a small CNN trained from scratch
  2. "mobilenet"       - transfer learning on top of MobileNetV2

Keep this file UI-free: app.py only ever calls the functions below.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils import CLASS_NAMES

IMG_SIZE = 160
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# A) Problem Setup helpers
# ---------------------------------------------------------------------------
def load_data(data_dir: str = "data"):
    """Load train/test image datasets from class-named sub-folders."""
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=False,
    )
    return train_ds, test_ds


def preprocess_data(dataset, augment: bool = False):
    """Normalize pixel values to [0,1] and optionally add light augmentation."""
    normalization = layers.Rescaling(1.0 / 255)

    if augment:
        aug = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
        ])
        dataset = dataset.map(lambda x, y: (aug(x, training=True), y))

    dataset = dataset.map(lambda x, y: (normalization(x), y))
    return dataset.prefetch(tf.data.AUTOTUNE)


def preprocess_single_image(pil_image):
    """Turn a PIL image into a normalized batch-of-1 tensor for prediction."""
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# B) Core Logic Module - model architectures
# ---------------------------------------------------------------------------
def build_baseline_cnn(num_classes: int = len(CLASS_NAMES)):
    """Small CNN trained from scratch. Acts as the comparison baseline."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mobilenet_model(num_classes: int = len(CLASS_NAMES)):
    """Transfer-learning model on top of MobileNetV2 (ImageNet weights)."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze for fast fine-tuning

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenet_v2_transfer")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, train_ds, epochs: int = 8, val_ds=None):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    return history


# ---------------------------------------------------------------------------
# Prediction (used live by the Streamlit app)
# ---------------------------------------------------------------------------
def run_model_or_algorithm(model, image_batch):
    """Return (predicted_class_name, confidence, full_probability_vector)."""
    probs = model.predict(image_batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


# ---------------------------------------------------------------------------
# E) Evaluation Module
# ---------------------------------------------------------------------------
def evaluate_model(model, test_ds):
    """Compute accuracy / precision / recall / F1 / confusion matrix on test_ds."""
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "class_names": CLASS_NAMES,
    }


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
