import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ====== تنظیمات اولیه ======
IMG_SIZE = 224
ethnic_labels = ['Arab', 'Iranian', 'IranianJews', 'Pashtun', 'Turkic']
iranian_labels = ['Baluch', 'Gilak', 'Hormozgani', 'Kurd', 'Lur', 'South_Khorasan', 'Yazdi']

# بارگذاری مدل‌ها
@st.cache_resource
def load_models():
    model = load_model("ethnicity_model.keras")
    model_irani = load_model("ethnicity_model_irani.keras")
    return model, model_irani

model, model_irani = load_models()

# ====== رابط کاربری ======
st.title("Ethnicity Recognition from Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # پیش‌پردازش تصویر
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_gray = img_resized.convert("L")
    arr = np.array(img_gray)
    arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0) / 255.0

    # پیش‌بینی
    preds = model.predict(arr)[0]
    preds_irani = model_irani.predict(arr)[0]

    # نمایش متن درصد قومیتی
    st.subheader("Predicted Ethnicity Percentages")
    result = dict(zip(ethnic_labels, preds.tolist()))
    result_irani = dict(zip(iranian_labels, preds_irani.tolist()))

    # نمایش Pie chart
    fig, ax = plt.subplots(figsize=(6,6))
    sizes = [v*100 for v in result.values()]
    ax.pie(sizes, labels=result.keys(), autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)
