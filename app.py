"""
Streamlit app – generates 1-10 MNIST-style digits using your trained cGAN
Run: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

LATENT_DIM   = 100
NUM_CLASSES  = 10
MODEL_FILE   = "generator_full.keras"   # <— same name you downloaded

# ---------- 1. Load generator only once per worker ----------
@st.cache_resource(show_spinner="Cargando modelo…")
def load_generator(model_path=MODEL_FILE):
    # load_model includes architecture, so no need to rebuild by hand
    return tf.keras.models.load_model(model_path, compile=False)

gen = load_generator()

# ---------- 2. Streamlit UI ----------
st.title("✍️ Generador de dígitos manuscritos (cGAN, 20 epochs)")
digit = st.number_input("Dígito (0-9)", min_value=0, max_value=9, value=4, step=1)
num   = 5

if st.button("Generar"):
    z   = tf.random.normal([num, LATENT_DIM])
    lbl = tf.constant([[digit]] * num)
    imgs = (gen([z, lbl], training=False) + 1) / 2      # scale [-1,1] → [0,1]
    cols = st.columns(num)
    for c, img in zip(cols, imgs.numpy().squeeze()):
        c.image(Image.fromarray((img * 255).astype("uint8"), "L"), use_column_width=True)
