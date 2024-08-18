import os
import mne
import pandas as pd

def edf_to_csv(edf_path, csv_path):
    if os.path.exists(csv_path):
        print(f"El archivo {csv_path} ya existe. Saltando conversión.")
        return
    print(f"Procesando archivo: {edf_path}")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        df = raw.to_data_frame()
        df.to_csv(csv_path, index=False)
        print(f"Archivo convertido y guardado como: {csv_path}")
    except Exception as e:
        print(f"Error procesando el archivo {edf_path}: {e}")

# Convertir archivos en carpetas 'healthy' y 'seizures'
base_path = '../../../../../data_EGG'
folders = ['healthy', 'seizures']

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.edf'):
                edf_path = os.path.join(folder_path, filename)
                csv_path = os.path.join(folder_path, filename.replace('.edf', '.csv'))
                edf_to_csv(edf_path, csv_path)
    else:
        print(f"La carpeta {folder_path} no existe. Por favor verifica la ruta.")
