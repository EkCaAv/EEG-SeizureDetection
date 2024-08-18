import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import mne

# Cargar el modelo entrenado
print("Cargando el modelo entrenado...")
model = load_model('best_model.h5')
print("Modelo cargado exitosamente.")

# Función para procesar el archivo .edf y extraer características
def process_edf(file_path):
    print(f"Procesando el archivo .edf: {file_path}")
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data = raw.get_data()
    print("Datos del archivo .edf cargados.")
    
    data = np.mean(data, axis=0)  # Promediar los canales para simplificar, ajustar según necesidad
    print("Datos promediados a lo largo de los canales.")
    
    data = data.reshape(1, -1)  # Reshape para cumplir con el input del modelo
    print("Datos reestructurados para el modelo.")
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    print("Datos normalizados.")
    
    return data

# Función para cargar el archivo y hacer la predicción
def load_file():
    print("Abriendo el cuadro de diálogo para seleccionar archivo .edf...")
    file_path = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf")])
    if file_path:
        print(f"Archivo seleccionado: {file_path}")
        
        data = process_edf(file_path)
        print("Haciendo predicción con el modelo...")
        prediction = model.predict(data)
        print(f"Predicción del modelo: {prediction}")
        
        result = 'Convulsión' if prediction >= 0.5 else 'No Convulsión'
        print(f"Resultado: {result}")
        
        messagebox.showinfo("Resultado de la Predicción", f"El archivo contiene: {result}")
    else:
        print("No se seleccionó ningún archivo.")

# Crear la interfaz
def create_interface():
    root = tk.Tk()
    root.title("Detección de Convulsiones en EEG")
    
    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)
    
    label = tk.Label(frame, text="Seleccione un archivo .edf para detectar convulsiones")
    label.pack(pady=10)
    
    button = tk.Button(frame, text="Cargar Archivo .edf", command=load_file)
    button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_interface()

# Función para evaluar el modelo
def evaluate_model(X_test, y_test):
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype(int)
    
    print("Evaluación del modelo:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
