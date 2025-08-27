import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

# مسیر فایل شمارنده
counter_file = "counter.txt"

# اگر فایل وجود ندارد، بساز و صفر را داخلش بنویس
if not os.path.exists(counter_file):
    with open(counter_file, "w") as f:
        f.write("0")

# خواندن شمارنده فعلی
with open(counter_file, "r") as f:
    upload_count = int(f.read().strip())

# ====== اپلیکیشن ======

st.title("🌍 سامانه تشخیص قومیت(Ethnicity Detection)")
st.write("یک تصویر پرتره آپلود کنید تا مدل نتایج پیش‌بینی را نمایش دهد.(Upload your portrait)")

uploaded_file = st.file_uploader("تصویر خود را آپلود کنید", type=["jpg", "jpeg", "png"])


# ====== تنظیمات ======
IMG_SIZE = 224
ethnic_labels = ['Arab', 'Iranian', 'IranianJews', 'Pashtun', 'Turkic']
iranian_labels = ['Baluch', 'Gilak', 'Hormozgani', 'Kurd', 'Lur', 'South_Khorasan', 'Yazdi']
colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0']

# ====== پیش‌پردازش ======
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img.convert("L")
    img_array = image.img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, Image.open(uploaded_file)

if uploaded_file is not None:
    # افزایش شمارنده
    upload_count += 1

    # ذخیره مجدد در فایل
    with open(counter_file, "w") as f:
        f.write(str(upload_count))
    st.image(uploaded_file, caption="تصویر ورودی (Uploaded Image)", use_container_width=True)
    st.success("✅ File uploaded successfully!")

    img_array, original_img = preprocess_image(uploaded_file)
    predictions = model.predict(img_array)[0]
    predictions_irani = model_irani.predict(img_array)[0]

    predictions_dict = dict(zip(ethnic_labels, predictions))
    predictions_irani_dict = dict(zip(iranian_labels, predictions_irani))

    st.subheader("🔹 نتایج پیش‌بینی گروه‌های قومی اصلی:")
    for k, v in predictions_dict.items():
        st.write(f"{k}: {v:.2%}")

    st.subheader("🔹 نتایج پیش‌بینی زیرگروه‌های ایرانی:")
    for k, v in predictions_irani_dict.items():
        st.write(f"{k}: {v:.2%}")

    st.subheader("📊 نمودار گروه‌های اصلی همراه با تصاویر")
    plot_ethnicity_pie(predictions_dict, prepared_images, original_img)


# نمایش تعداد آپلودها

# ------------------------------
# نمایش تعداد آپلودها در کارت
# ------------------------------
card_html = f"""
<div style="
    background: rgba(0, 123, 255, 0.15); 
    border-radius: 12px; 
    padding: 15px; 
    width: 250px; 
    text-align: center;
    margin-bottom: 20px;
    font-family: sans-serif;
">
    <h4 style="margin: 0; color: #007bff;">📊 Total Uploads</h4>
    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #000;">{upload_count}</p>
</div>
"""

st.markdown(card_html, unsafe_allow_html=True)



# 🌄 پس‌زمینه کل صفحه + شفافیت + گردی گوشه‌ها
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/Goharshadi-m/ethnicity-app/main/header.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* حذف هدر سفید بالای صفحه */
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.5); /* سایدبار نیمه شفاف */
}

.block-container {
    background-color: rgba(255, 255, 255, 0.75); 
    border-radius: 15px;
    padding: 20px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        background: rgba(255, 255, 255, 0.5);
        padding: 10px;
        border-radius: 12px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        color: #333;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }
    .footer a {
        color: #0066cc;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.3s ease;
    }
    .footer a:hover {
        color: #ff6600;
    }
    </style>

    <div class="footer">
        This project was developed by <b>Mostafa Goharshadi</b>.<br>
        For improvements or similar projects, feel free to contact me on 
        <a href="https://wa.me/989304441138" target="_blank">WhatsApp</a>.
    </div>
    """,
    unsafe_allow_html=True
)

# ====== بارگذاری مدل‌ها ======
@st.cache_resource
def load_models():
    model = load_model("ethnicity_model.keras")
    model_irani = load_model("ethnicity_model_irani.keras")
    return model, model_irani

model, model_irani = load_models()

# ====== بارگذاری عکس‌های اقوام ======
def load_ethnic_images():
    prepared_images = {}
    target_size = (100, 100)  # 👈 همه عکس‌ها یک اندازه ثابت
    for label in ethnic_labels:
        img_path = f"{label}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGBA")
            img = img.resize(target_size, Image.Resampling.LANCZOS)  # 👈 تغییر اینجا
            prepared_images[label] = img
    return prepared_images

prepared_images = load_ethnic_images()



# ====== رسم نمودار ======
def plot_ethnicity_pie(predictions_dict, prepared_images, center_img):
    labels = list(predictions_dict.keys())
    sizes = [predictions_dict[k] * 100 for k in labels]

    # اصلاح سایز wedge ها
    plot_sizes = []
    for size in sizes:
        if size < 10 and size > 0:
            plot_sizes.append(10)
        elif size == 0:
            plot_sizes.append(0)
        else:
            plot_sizes.append(size)

    total_plot_size = sum(plot_sizes)
    if total_plot_size > 0:
        plot_sizes = [s / total_plot_size * 100 for s in plot_sizes]
    else:
        plot_sizes = [0] * len(sizes)

    # رسم نمودار
    fig, ax = plt.subplots(figsize=(8, 8))
    wedgeprops = {'width': 0.4}
    wedges, texts = ax.pie(plot_sizes, labels=None, colors=colors, startangle=140, wedgeprops=wedgeprops)

    # اضافه کردن برچسب و تصویر هر قوم
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2.
        radius = 1.1
        x = radius * np.cos(np.deg2rad(angle))
        y = radius * np.sin(np.deg2rad(angle))

        original_percentage = sizes[i]
        label = labels[i]

        if original_percentage > 0:
            ax.text(x, y, f"{label}: {original_percentage:.1f}%", ha='center', va='center', fontsize=9)

        if label in prepared_images and original_percentage > 0:
            img_to_add = prepared_images[label]
            imagebox_inside = OffsetImage(img_to_add, zoom=0.3)
            inner_radius = 1 - wedgeprops['width']
            outer_radius = 1
            image_radius_position = (inner_radius + outer_radius) / 2.0
            x_img = image_radius_position * np.cos(np.deg2rad(angle))
            y_img = image_radius_position * np.sin(np.deg2rad(angle))
            ab_inside = AnnotationBbox(imagebox_inside, (x_img, y_img), frameon=False, pad=0.0)
            ax.add_artist(ab_inside)

    # تصویر مرکزی
    if center_img is not None:
        inner_hole_diameter = 1 - wedgeprops['width']
        img_size_for_center = int(plt.rcParams['figure.figsize'][0] * fig.dpi * inner_hole_diameter * 0.7)
        center_img_resized = center_img.resize((img_size_for_center, img_size_for_center), Image.Resampling.LANCZOS)

        # ماسک دایره‌ای
        mask = Image.new('L', (img_size_for_center, img_size_for_center), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img_size_for_center, img_size_for_center), fill=255)
        center_img_resized.putalpha(mask)

        imagebox_center = OffsetImage(center_img_resized, zoom=1)
        ab_center = AnnotationBbox(imagebox_center, (0, 0), frameon=False, pad=0)
        ax.add_artist(ab_center)

    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)















