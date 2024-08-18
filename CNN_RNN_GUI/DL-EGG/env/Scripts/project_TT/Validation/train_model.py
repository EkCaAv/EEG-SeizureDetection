import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import mne

# Funciones para cargar y procesar datos
def align_dataframes(df1, df2):
    common_columns = df1.columns.intersection(df2.columns)
    return df1[common_columns], df2[common_columns]

def load_and_process_data(file_path, label, sample_size=100000):
    df = pd.read_csv(file_path)
    df['label'] = label
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

# Rutas de archivos CSV
healthy_train_path = '../../../../data_EGG/healthy/cleaned_data_train.csv'
seizures_train_path = '../../../../data_EGG/seizures/cleaned_data_train.csv'
healthy_test_path = '../../../../data_EGG/healthy/cleaned_data_test.csv'
seizures_test_path = '../../../../data_EGG/seizures/cleaned_data_test.csv'

# Cargar y procesar datos de entrenamiento y prueba
X_train_healthy, y_train_healthy = load_and_process_data(healthy_train_path, label=0)
X_train_seizures, y_train_seizures = load_and_process_data(seizures_train_path, label=1)
X_test_healthy, y_test_healthy = load_and_process_data(healthy_test_path, label=0)
X_test_seizures, y_test_seizures = load_and_process_data(seizures_test_path, label=1)

# Alinear DataFrames
X_train_healthy_df, X_train_seizures_df = align_dataframes(pd.DataFrame(X_train_healthy), pd.DataFrame(X_train_seizures))
X_train_healthy = X_train_healthy_df.values
X_train_seizures = X_train_seizures_df.values
X_test_healthy_df, X_test_seizures_df = align_dataframes(pd.DataFrame(X_test_healthy), pd.DataFrame(X_test_seizures))
X_test_healthy = X_test_healthy_df.values
X_test_seizures = X_test_seizures_df.values

# Concatenar datos
X_train = np.concatenate((X_train_healthy, X_train_seizures), axis=0)
y_train = np.concatenate((y_train_healthy, y_train_seizures), axis=0)
X_test = np.concatenate((X_test_healthy, X_test_seizures), axis=0)
y_test = np.concatenate((y_test_healthy, y_test_seizures), axis=0)

# Normalizar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir etiquetas a categorías binarias
y_train = np.array(y_train)
y_test = np.array(y_test)

# Cargar el mejor modelo
best_model_path = 'best_model.h5'
best_model = load_model(best_model_path)

# Evaluar el mejor modelo en el conjunto de prueba
y_pred = (best_model.predict(X_test) > 0.5).astype("int32")

# Generar matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Generar informe de clasificación
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Calcular métricas adicionales
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Función para procesar el archivo .edf y extraer características
def process_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data = raw.get_data()
    data = np.mean(data, axis=0)  # Promediar los canales para simplificar, ajustar según necesidad
    data = data.reshape(1, -1)  # Reshape para cumplir con el input del modelo
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

# Función para cargar el archivo y hacer la predicción
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf")])
    if file_path:
        data = process_edf(file_path)
        prediction = best_model.predict(data)
        result = 'Convulsión' if prediction >= 0.6 else 'No Convulsión'
        messagebox.showinfo("Resultado de la Predicción", f"El archivo contiene: {result}")

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
