# CropGuard AI
Bilingual (English/Urdu) Crop Disease Detection System — AI Fundamentals Lab Project
(Track: Option A — FYP Integration)

## Problem
Farmers in rural Sindh often lack quick access to crop disease diagnosis. CropGuard AI
takes a photo of a crop leaf and identifies whether it is healthy or affected by one of
three common diseases (Leaf Blight, Powdery Mildew, Leaf Rust), returning a bilingual
diagnosis with a confidence score, a visual explanation, and a treatment recommendation.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get a dataset

This repo ships with a script to generate a small **synthetic placeholder dataset**
so the whole pipeline runs immediately without internet or a large download:

```bash
python generate_sample_data.py
```

This creates `data/train/<class>/...` and `data/test/<class>/...` with 4 classes:
`Healthy`, `Leaf_Blight`, `Powdery_Mildew`, `Leaf_Rust`.

**Before your viva**, replace the contents of `data/train` and `data/test` with real
labelled leaf photographs (e.g. PlantVillage dataset) using the same folder structure —
the model and app do not need any code changes, only real images in the same folders.

## 2. Train both models

```bash
python train.py
```

This trains:
- `baseline_cnn` — a small CNN trained from scratch (comparison baseline)
- `mobilenet` — MobileNetV2 transfer learning (main model)

and saves both models plus `saved_models/metrics.json` (accuracy/precision/recall/F1
for both, used by the Evaluation tab).

> Training MobileNetV2 requires internet access the *first* time, to download
> ImageNet weights automatically via Keras.

## 3. Run the app

```bash
streamlit run app.py
```

## Project structure

```
CropGuardAI/
├── app.py                  # Streamlit UI (render_ui)
├── model.py                 # load_data, preprocess_data, build models, train, evaluate
├── explain.py                # Grad-CAM + bilingual natural-language explanations
├── utils.py                  # bilingual labels / treatment text
├── generate_sample_data.py   # creates offline synthetic sample dataset
├── train.py                  # trains + evaluates both models
├── requirements.txt
├── data/                     # train/ and test/ image folders (per class)
├── saved_models/             # trained .keras models + metrics.json (created after training)
└── screenshots/              # add your UI screenshots here before submission
```

## How the 5 required lab modules map to this project

| Module | Where |
|---|---|
| A) Problem Setup | Sidebar controls + file uploader/sample picker in `app.py`, input validation |
| B) Core Logic | `model.py` — baseline CNN + MobileNetV2 transfer learning |
| C) Visual UI | Confidence bar chart, Grad-CAM heatmap, tabs, result metric cards |
| D) Explainability | `explain.py` — Grad-CAM heatmap + bilingual natural-language explanation |
| E) Evaluation | "Evaluation" tab — accuracy/precision/recall/F1, baseline vs. MobileNetV2 comparison chart |

## Limitations & future improvements
- Sample dataset is synthetic; a real, larger, field-collected dataset would improve
  accuracy and generalization.
- Only 4 classes covered; can be extended to more crop-specific diseases.
- Urdu translations are currently static templates rather than a full NLP-generated
  explanation (could add an LLM-based explanation layer as an extension — see Lab
  Guide Option 4).

## Viva prep notes
- **Why this problem**: crop disease misdiagnosis directly affects yields for small
  farmers in rural Sindh (author's family background in agriculture-adjacent business).
- **Why MobileNetV2**: lightweight enough to eventually run on a phone in the field,
  good accuracy/speed trade-off, works well with transfer learning on small datasets.
- **Data & processing**: images resized to 160×160, normalized to [0,1], augmented
  (flip/rotate/zoom) during training.
- **How UI helps**: confidence chart + Grad-CAM heatmap let a non-technical user see
  *what* the model is confident about and *where* it looked, not just a raw label.
- **AI component**: CNN image classification (core), Grad-CAM (explainability).
- **Limitations**: synthetic demo data, limited class count, static bilingual text.
