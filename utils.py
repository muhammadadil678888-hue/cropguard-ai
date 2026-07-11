"""
utils.py
Bilingual (English/Urdu) text resources and small shared helpers for CropGuard AI.
"""

CLASS_NAMES = ["Healthy", "Leaf_Blight", "Powdery_Mildew", "Leaf_Rust"]

# Bilingual display names
CLASS_LABELS = {
    "Healthy":         {"en": "Healthy",        "ur": "صحت مند"},
    "Leaf_Blight":     {"en": "Leaf Blight",     "ur": "پتوں کا جھلساؤ"},
    "Powdery_Mildew":  {"en": "Powdery Mildew",  "ur": "پاؤڈری پھپھوندی"},
    "Leaf_Rust":       {"en": "Leaf Rust",       "ur": "پتوں کا زنگ"},
}

# Short bilingual treatment recommendations
TREATMENT_ADVICE = {
    "Healthy": {
        "en": "No disease detected. Continue regular monitoring and irrigation schedule.",
        "ur": "کوئی بیماری نہیں ملی۔ باقاعدہ نگرانی اور آبپاشی جاری رکھیں۔",
    },
    "Leaf_Blight": {
        "en": "Remove affected leaves and apply a copper-based fungicide. Avoid overhead watering.",
        "ur": "متاثرہ پتے ہٹا دیں اور کاپر پر مبنی فنجی سائیڈ استعمال کریں۔ اوپر سے پانی دینے سے گریز کریں۔",
    },
    "Powdery_Mildew": {
        "en": "Improve air circulation and apply sulfur-based fungicide spray weekly.",
        "ur": "ہوا کی گردش بہتر بنائیں اور ہفتہ وار سلفر پر مبنی سپرے کریں۔",
    },
    "Leaf_Rust": {
        "en": "Apply recommended fungicide and remove volunteer plants that host the rust fungus.",
        "ur": "تجویز کردہ فنجی سائیڈ استعمال کریں اور زنگ پھیلانے والے جنگلی پودے ہٹا دیں۔",
    },
}

UI_TEXT = {
    "title": {"en": "CropGuard AI", "ur": "کراپ گارڈ اے آئی"},
    "subtitle": {
        "en": "Bilingual Crop Disease Detection System",
        "ur": "دو لسانی فصل کی بیماری کی شناخت کا نظام",
    },
    "upload_prompt": {
        "en": "Upload a leaf image or choose a sample below",
        "ur": "پتے کی تصویر اپلوڈ کریں یا نیچے دیا گیا نمونہ منتخب کریں",
    },
    "run_button": {"en": "Run Diagnosis", "ur": "تشخیص چلائیں"},
    "result_header": {"en": "Diagnosis Result", "ur": "تشخیص کا نتیجہ"},
    "explain_header": {"en": "Why this result?", "ur": "یہ نتیجہ کیوں؟"},
    "eval_header": {"en": "Model Evaluation", "ur": "ماڈل کی کارکردگی"},
}


def t(key: str, lang: str) -> str:
    """Return UI_TEXT[key] in the requested language ('en' or 'ur')."""
    return UI_TEXT.get(key, {}).get(lang, key)


def class_label(class_name: str, lang: str) -> str:
    return CLASS_LABELS.get(class_name, {}).get(lang, class_name)


def treatment_for(class_name: str, lang: str) -> str:
    return TREATMENT_ADVICE.get(class_name, {}).get(lang, "")
