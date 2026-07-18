"""
app.py
CropGuard AI - Streamlit front-end.
Run:  streamlit run app.py
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from model import preprocess_single_image, run_model_or_algorithm, load_metrics
from explain import grad_cam_heatmap, generate_explanation
from utils import CLASS_NAMES, class_label, t

MODEL_DIR = "saved_models"
SAMPLE_DIR = os.path.join("data", "test")

st.set_page_config(page_title="CropGuard AI", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(name: str):
    from model import build_baseline_cnn, build_mobilenet_model
    if name == "baseline_cnn":
        model = build_baseline_cnn()
    else:
        model = build_mobilenet_model()
    ckpt_dir = os.path.join(MODEL_DIR, f"{name}_ckpt")
    if not os.path.exists(ckpt_dir):
        return None
    model.load_weights(os.path.join(ckpt_dir, "ckpt"))
    return model
    


@st.cache_data
def get_sample_images():
    samples = {}
    for cls in CLASS_NAMES:
        files = glob.glob(os.path.join(SAMPLE_DIR, cls, "*.png"))
        if files:
            samples[cls] = files[0]
    return samples


# ---------------------------------------------------------------------------
# Sidebar: language + model + controls  (Problem Setup Module)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    lang = st.radio("Language / زبان", options=["en", "ur"],
                     format_func=lambda x: "English" if x == "en" else "اردو")
    model_choice = st.selectbox(
        "Model", options=["mobilenet", "baseline_cnn"],
        format_func=lambda x: "MobileNetV2 (transfer learning)" if x == "mobilenet"
        else "Baseline CNN (from scratch)",
    )
    st.caption("Compare both models in the Evaluation tab below.")

st.title(f"🌾 {t('title', lang)}")
st.caption(t("subtitle", lang))

model = load_trained_model(model_choice)

if model is None:
    st.error(
        "No trained model found in `saved_models/`. Run `python generate_sample_data.py` "
        "then `python train.py` first (see README.md)."
    )
    st.stop()

tab_diagnose, tab_eval = st.tabs(["🔍 Diagnose", f"📊 {t('eval_header', lang)}"])

# ---------------------------------------------------------------------------
# TAB 1 — Problem Setup + Core Logic + Visual UI + Explainability
# ---------------------------------------------------------------------------
with tab_diagnose:
    st.subheader(t("upload_prompt", lang))

    col_input, col_samples = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    with col_samples:
        samples = get_sample_images()
        sample_choice = st.selectbox(
            "...or pick a sample", options=["None"] + list(samples.keys())
        )

    # --- Input validation & resolution ---
    pil_image = None
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
        except Exception:
            st.error("⚠️ Could not read that file. Please upload a valid JPG/PNG image.")
    elif sample_choice != "None":
        pil_image = Image.open(samples[sample_choice])

    if pil_image is not None:
        img_col, result_col = st.columns([1, 1.3])

        with img_col:
            st.image(pil_image, caption="Input image", use_container_width=True)
            run_clicked = st.button(f"▶️ {t('run_button', lang)}", type="primary")

        if run_clicked:
            with st.spinner("Running diagnosis..."):
                batch = preprocess_single_image(pil_image)
                pred_class, confidence, probs = run_model_or_algorithm(model, batch)
                class_idx = CLASS_NAMES.index(pred_class)
                heatmap = grad_cam_heatmap(model, batch, class_idx)
                explanation_text = generate_explanation(pred_class, confidence, probs, lang)

            with result_col:
                st.success(f"✅ {t('result_header', lang)}")
                st.metric(
                    label=class_label(pred_class, lang),
                    value=f"{confidence * 100:.1f}% confidence",
                )

                # Visual UI: confidence bar chart across all classes
                df = pd.DataFrame({
                    "Disease": [class_label(c, lang) for c in CLASS_NAMES],
                    "Confidence": probs,
                })
                fig, ax = plt.subplots(figsize=(5, 2.8))
                colors = ["#2e7d32" if c == pred_class else "#90a4ae" for c in CLASS_NAMES]
                ax.barh(df["Disease"], df["Confidence"], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Confidence")
                st.pyplot(fig)

            # Explainability panel (full width)
            st.markdown(f"### 🔎 {t('explain_header', lang)}")
            exp_col1, exp_col2 = st.columns([1, 1.3])
            with exp_col1:
                if heatmap is not None:
                    fig2, ax2 = plt.subplots(figsize=(3, 3))
                    ax2.imshow(pil_image.resize((160, 160)))
                    ax2.imshow(heatmap, cmap="jet", alpha=0.45)
                    ax2.axis("off")
                    ax2.set_title("Grad-CAM: influential regions")
                    st.pyplot(fig2)
                else:
                    st.info("Visual heatmap unavailable for this model architecture.")
            with exp_col2:
                st.write(explanation_text)
                st.progress(float(confidence))
    else:
        st.info(t("upload_prompt", lang))

# ---------------------------------------------------------------------------
# TAB 2 — E) Evaluation Module (compare baseline vs MobileNetV2)
# ---------------------------------------------------------------------------
with tab_eval:
    metrics = load_metrics(os.path.join(MODEL_DIR, "metrics.json"))
    if metrics is None:
        st.warning("No metrics.json found. Run `python train.py` to generate evaluation results.")
    else:
        st.subheader("Baseline CNN vs. MobileNetV2 (transfer learning)")

        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": "Baseline CNN" if name == "baseline_cnn" else "MobileNetV2",
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-score": m["f1"],
            })
        df_metrics = pd.DataFrame(rows)
        st.dataframe(df_metrics.style.highlight_max(axis=0, subset=df_metrics.columns[1:],
                                                      color="#c8e6c9"),
                     use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(df_metrics))
        width = 0.2
        for i, col in enumerate(["Accuracy", "Precision", "Recall", "F1-score"]):
            ax3.bar(x + i * width, df_metrics[col], width, label=col)
        ax3.set_xticks(x + 1.5 * width)
        ax3.set_xticklabels(df_metrics["Model"])
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.set_title("Model comparison")
        st.pyplot(fig3)

        st.caption(
            "Metrics computed on the held-out test split. Confusion matrices are stored "
            "in saved_models/metrics.json for further inspection."
        )
