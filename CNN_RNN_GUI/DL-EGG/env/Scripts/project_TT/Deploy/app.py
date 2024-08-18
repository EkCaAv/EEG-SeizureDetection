import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from PIL import Image
from tensorflow.keras.models import load_model
import io
import tempfile
import os

# Funciones para procesar el archivo .edf y convertirlo en imagen
def read_edf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    raw = mne.io.read_raw_edf(tmp_path, preload=True)
    os.remove(tmp_path)  # Eliminar el archivo temporal después de su uso
    return raw

def create_spectrogram(data, fs=256, nperseg=256):
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=nperseg)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Convert to dB scale for better visualization
    return Sxx_db

def save_spectrogram_image(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(spectrogram, aspect='auto', cmap='plasma', origin='lower')
    fig.colorbar(cax, ax=ax, label='Intensity (dB)')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    
    # Save to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# Cargar el modelo entrenado
model = load_model('best_cnn_model.h5')

# Interfaz Streamlit
st.title('Clasificador de EEG')

uploaded_file = st.file_uploader("Elige un archivo .edf", type="edf")

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    try:
        # Leer el archivo .edf
        raw = read_edf(uploaded_file)
        st.write("File read successfully.")
        
        # Convertir los datos del primer canal a espectrograma
        spectrogram = create_spectrogram(raw.get_data()[0])  # Obtener los datos del primer canal
        
        # Guardar el espectrograma como imagen
        img_buf = save_spectrogram_image(spectrogram)
        st.image(img_buf, caption='Espectrograma del archivo .edf', use_column_width=True)

        # Preprocesar la imagen para la predicción
        img = Image.open(img_buf).convert('RGB')  # Convertir a RGB para asegurar que tiene 3 canales
        img = img.resize((128, 128))  # Asegúrate de que el tamaño sea el correcto
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión del batch

        # Realizar la predicción
        prediction = model.predict(img_array)
        class_label = 'Convulsión' if prediction[0] > 0.5 else 'No convulsión'
        
        st.write(f'Predicción: {class_label}')
    except Exception as e:
        st.write(f"An error occurred: {e}")
