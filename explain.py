"""
explain.py
D) Explainability Module for CropGuard AI.

Provides:
  - grad_cam_heatmap(): visual explanation (which pixels influenced the decision)
  - generate_explanation(): short bilingual natural-language explanation
"""

import numpy as np
import tensorflow as tf

from utils import class_label, treatment_for


def _find_last_conv_layer(model):
    """Locate the last convolutional layer for Grad-CAM (works for both
    the baseline CNN and the MobileNetV2 wrapper)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # nested MobileNetV2 base model - search inside it
            for sub in reversed(layer.layers):
                if len(sub.output_shape) == 4:
                    return layer, sub.name
        if len(getattr(layer, "output_shape", [])) == 4:
            return model, layer.name
    return None, None


def grad_cam_heatmap(model, image_batch, class_index):
    """Return a (H, W) heatmap in [0,1] highlighting regions that most
    influenced the predicted class, or None if it can't be computed."""
    try:
        container, layer_name = _find_last_conv_layer(model)
        if container is None:
            return None

        grad_model = tf.keras.models.Model(
            [container.inputs if hasattr(container, "inputs") else model.input],
            [container.get_layer(layer_name).output, container.output
             if container is not model else model.output],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(image_batch)
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        # Grad-CAM is a "nice to have" visual; never crash the app over it.
        return None


def generate_explanation(class_name: str, confidence: float, probs, lang: str = "en"):
    """Produce a short natural-language explanation of the result."""
    label = class_label(class_name, lang)
    advice = treatment_for(class_name, lang)
    pct = round(confidence * 100, 1)

    if lang == "ur":
        text = (
            f"ماڈل نے {pct}% اعتماد کے ساتھ '{label}' کی تشخیص کی، "
            f"جو کہ پتے کے رنگ، بناوٹ اور دھبوں کی بنیاد پر کی گئی۔ "
            f"مشورہ: {advice}"
        )
    else:
        text = (
            f"The model diagnosed '{label}' with {pct}% confidence, based on "
            f"leaf color, texture, and spot patterns detected in the image. "
            f"Recommendation: {advice}"
        )
    return text
