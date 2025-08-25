import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# ====== تنظیمات اولیه ======
IMG_SIZE = 224
ethnic_labels = ['Arab', 'Iranian', 'IranianJews', 'Pashtun', 'Turkic']
iranian_labels = ['Baluch', 'Gilak', 'Hormozgani', 'Kurd', 'Lur', 'South_Khorasan', 'Yazdi']

# ====== بارگذاری مدل‌ها ======
@st.cache_resource
def load_models():
    model = load_model("ethnicity_model.keras")
    model_irani = load_model("ethnicity_model_irani.keras")
    return model, model_irani

model, model_irani = load_models()

# ====== پیش‌پردازش ======
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img.convert("L")  # تبدیل به grayscale
    img_array = image.img_to_array(img)  # شکل (224,224,1)
    img_array = np.repeat(img_array, 3, axis=-1)  # تبدیل به 3 کاناله
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ====== رسم نمودار ======
def plot_ethnicity_pie(predictions_dict):
    fig, ax = plt.subplots()
    ax.pie(predictions_dict.values(), labels=predictions_dict.keys(), autopct='%1.1f%%')
    ax.set_title("Ethnicity Prediction Distribution")
    st.pyplot(fig)

# ====== اپلیکیشن Streamlit ======
st.title("🌍 Ethnicity & Iranian Subgroups Classifier")

uploaded_file = st.file_uploader("یک تصویر بارگذاری کنید", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="تصویر ورودی", use_container_width=True)

    # پردازش تصویر
    img_array = preprocess_image(uploaded_file)

    # پیش‌بینی
    predictions = model.predict(img_array)[0]
    predictions_irani = model_irani.predict(img_array)[0]

    # نتایج به صورت دیکشنری
    predictions_dict = dict(zip(ethnic_labels, predictions))
    predictions_irani_dict = dict(zip(iranian_labels, predictions_irani))

    # نمایش نتایج
    st.subheader("🔹 نتایج پیش‌بینی گروه‌های قومی اصلی:")
    for k, v in predictions_dict.items():
        st.write(f"{k}: {v:.2%}")

    st.subheader("🔹 نتایج پیش‌بینی زیرگروه‌های ایرانی:")
    for k, v in predictions_irani_dict.items():
        st.write(f"{k}: {v:.2%}")

    # نمایش نمودار
    st.subheader("📊 نمودار پیش‌بینی گروه‌های اصلی")
    plot_ethnicity_pie(predictions_dict)
