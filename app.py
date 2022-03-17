import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.image(
    "logo.png", width=50, use_column_width=50)
st.title('Tanidok')

st.markdown("Tanidok merupakan aplikasi berbasis website yang menggunakan teknologi machine learning dalam mendeteksi penyakit tanaman. Adapun jenis tanaman yang dapat diteksi oleh aplikasi Tanidok ini adalah 14 tanaman. Tanaman tersebut adalah apel, ceri, jagung, anggur, persik, pepper, kentang, stroberi, tomat, blueberry, jeruk, kedelai, raspberry, dan squash.")


def main():
    file_uploaded = st.file_uploader(
        "Unggah Gambar Daun", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Terjadi kesalahan, upload file berupa gambar")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "base_dir.h5"
    IMAGE_SHAPE = (224, 224, 3)
    model = load_model(classifier_model, compile=False,
                       custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224, 224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'


    ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    result = f"Hasil klasifikasi penyakit tanaman: {class_names[np.argmax(scores)]}"
    return result


if __name__ == "__main__":
    main()
