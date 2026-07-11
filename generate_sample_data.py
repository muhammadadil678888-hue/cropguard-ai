"""
generate_sample_data.py

Creates a small SAMPLE leaf-image dataset so the app can run end-to-end
without internet access or a large real dataset.

IMPORTANT: These are synthetic placeholder images (colored/textured patterns),
NOT real crop photographs. Replace the contents of data/train and data/test
with your real labelled leaf images before your final demo/viva -- the folder
structure (one sub-folder per class) is what matters, the model does not care
whether the images are synthetic while you are testing the pipeline.

Usage:
    python generate_sample_data.py
"""

import os
import random
from PIL import Image, ImageDraw, ImageFilter

from utils import CLASS_NAMES

IMG_SIZE = 160
IMAGES_PER_CLASS_TRAIN = 40
IMAGES_PER_CLASS_TEST = 10

# Base colour + noise style per class so a real model can learn something
# minimally separable (green & clean = healthy, brown patches = blight, etc.)
CLASS_STYLE = {
    "Healthy":        {"base": (60, 140, 60),  "spot_color": None,          "spots": 0},
    "Leaf_Blight":    {"base": (70, 120, 50),  "spot_color": (120, 70, 30), "spots": 14},
    "Powdery_Mildew": {"base": (80, 130, 70),  "spot_color": (235, 235, 220), "spots": 20},
    "Leaf_Rust":      {"base": (90, 110, 40),  "spot_color": (170, 90, 20), "spots": 18},
}


def make_leaf_image(class_name: str) -> Image.Image:
    style = CLASS_STYLE[class_name]
    base = style["base"]
    # random small jitter so images in the same class aren't identical
    jitter = lambda c: max(0, min(255, c + random.randint(-15, 15)))
    bg = tuple(jitter(c) for c in base)

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    # simple leaf-vein lines
    draw.line([(IMG_SIZE // 2, 0), (IMG_SIZE // 2, IMG_SIZE)], fill=(40, 90, 40), width=2)
    for y in range(10, IMG_SIZE, 20):
        draw.line([(IMG_SIZE // 2, y), (IMG_SIZE // 2 - 20, y - 10)], fill=(40, 90, 40), width=1)
        draw.line([(IMG_SIZE // 2, y), (IMG_SIZE // 2 + 20, y - 10)], fill=(40, 90, 40), width=1)

    # disease spots
    if style["spots"]:
        for _ in range(style["spots"]):
            x = random.randint(5, IMG_SIZE - 10)
            y = random.randint(5, IMG_SIZE - 10)
            r = random.randint(3, 7)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=style["spot_color"])

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def build_dataset(root: str = "data"):
    for split, n in (("train", IMAGES_PER_CLASS_TRAIN), ("test", IMAGES_PER_CLASS_TEST)):
        for class_name in CLASS_NAMES:
            out_dir = os.path.join(root, split, class_name)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                img = make_leaf_image(class_name)
                img.save(os.path.join(out_dir, f"{class_name}_{i:03d}.png"))
    print(f"Sample dataset created under '{root}/train' and '{root}/test' "
          f"({len(CLASS_NAMES)} classes).")


if __name__ == "__main__":
    build_dataset()
