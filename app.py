import os
import keras
from tensorflow.keras.models import load_model
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# Load external CSS file
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load CSS
load_css("style.css")

st.header("Ultrasound Classification using CNN")

type_name = ['infected', 'notinfected']

model = load_model('Medical.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The ultrasound image reveals that the tissue is ' + type_name[np.argmax(result)] + ' with a score of ' + str(np.max(result)*100)+'%'
    return outcome

uploaded_file = st.file_uploader("Upload an image...", type="jpg")
file_path = None
if uploaded_file is not None:
    file_path = os.path.join(uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Create a layout with two columns (image on the left, text on the right)
    col1, col2 = st.columns([3.5, 8])  # Ratio of 1:2 for image and text

    with col1:
        # Set image size to a smaller width of 150px
        st.image(file_path, width=150, use_container_width=True)

    with col2:
        result = classify_images(file_path)
        st.markdown(f"<div class='result-text'>{result}</div>", unsafe_allow_html=True)


