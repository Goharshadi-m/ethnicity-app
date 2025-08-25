import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tensorflow.keras.preprocessing import image

# ============ پیش‌پردازش ============
def preprocess_image(img_path, IMG_SIZE=224):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # 🔹 همان چیزی که گفتی:
    img = img.convert('L')   # grayscale
    img_array = image.img_to_array(img)  # (224,224,1)
    img_array = np.repeat(img_array, 3, axis=-1)  # (224,224,3)
    img_array = np.expand_dims(img_array, axis=0) # (1,224,224,3)
    img_array /= 255.0

    return img_array

# ============ آماده‌سازی متن خروجی ============
def format_ethnicity_output(preds: dict, iranian_subpreds: dict):
    preds = {k: v * 100 for k, v in preds.items()}
    iranian_subpreds = {k: v * 100 for k, v in iranian_subpreds.items()}
    total_iranian = preds.get('Iranian', 0)
    sum_subs = sum(iranian_subpreds.values())

    if sum_subs > 0:
        normalized_subs = {k: round(v * total_iranian / sum_subs, 2) for k, v in iranian_subpreds.items()}
    else:
        normalized_subs = {k: 0.0 for k in iranian_subpreds}

    iranian_str = f"Iranian: {total_iranian:.2f}% (" + ', '.join([f"{k}: {v:.2f}%" for k, v in normalized_subs.items()]) + ")"
    other_groups = [k for k in preds if k != 'Iranian']
    others_str = '\n'.join([f"{k}: {preds[k]:.2f}%" for k in other_groups])

    return iranian_str + '\n' + others_str

# ============ آماده‌سازی عکس‌های اقوام ============
def prepare_ethnic_images():
    ethnic_groups = ["Arab", "Iranian", "IranianJews", "Pashtun", "Turkic"]
    prepared_images = {}
    size = (120, 120)

    for group in ethnic_groups:
        img_file = f"{group}.jpg"
        if os.path.exists(img_file):
            img = Image.open(img_file).convert("RGB")
            img_resized = img.resize(size, Image.Resampling.LANCZOS)

            # دایره‌ای کردن
            mask = Image.new('L', size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0) + size, fill=255)
            img_resized.putalpha(mask)

            prepared_images[group] = img_resized
    return prepared_images

# ============ رسم نمودار پای ============
def plot_ethnicity_pie(predictions_dict, prepared_images):
    labels = list(predictions_dict.keys())
    sizes = [predictions_dict[k] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedgeprops = {'width': 0.4}
    wedges, texts = ax.pie(sizes, startangle=140, wedgeprops=wedgeprops)

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2.
        radius = 1.1
        x = radius * np.cos(np.deg2rad(angle))
        y = radius * np.sin(np.deg2rad(angle))
        percent = sizes[i]
        label = labels[i]

        if percent > 0:
            ax.text(x, y, f"{label}: {percent:.1f}%", ha='center', va='center', fontsize=9)

        if label in prepared_images and percent > 0:
            imagebox = OffsetImage(prepared_images[label], zoom=0.45)
            inner_radius = 1 - wedgeprops['width']
            outer_radius = 1
            img_radius = (inner_radius + outer_radius) / 2.0
            x_img = img_radius * np.cos(np.deg2rad(angle))
            y_img = img_radius * np.sin(np.deg2rad(angle))
            ab = AnnotationBbox(imagebox, (x_img, y_img), frameon=False, pad=0.0)
            ax.add_artist(ab)

    ax.axis('equal')
    plt.tight_layout()
    plt.show()

# ============ نمونه استفاده ============
"""
img_array = preprocess_image("test.jpg")

predictions = model.predict(img_array)
predictions_irani = model_irani.predict(img_array)

predictions_dict = dict(zip(ethnic_labels, predictions[0]))
predictions_irani_dict = dict(zip(iranian_labels, predictions_irani[0]))

print(format_ethnicity_output(predictions_dict, predictions_irani_dict))

prepared_images = prepare_ethnic_images()
plot_ethnicity_pie(predictions_dict, prepared_images)
"""


