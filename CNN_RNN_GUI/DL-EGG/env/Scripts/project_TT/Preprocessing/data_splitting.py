import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csv_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Guardar los conjuntos de datos divididos
    train_csv_path = csv_path.replace('.csv', '_train.csv')
    test_csv_path = csv_path.replace('.csv', '_test.csv')
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f'Data split completed. Train data saved to {train_csv_path}, Test data saved to {test_csv_path}')

# Rutas de archivos CSV de ejemplo
healthy_csv = '../../../../data_EGG/healthy/cleaned_data.csv'
seizures_csv = '../../../../data_EGG/seizures/cleaned_data.csv'

# Verificar si los archivos existen antes de proceder
if os.path.exists(healthy_csv):
    split_data(healthy_csv)
else:
    print(f'File not found: {healthy_csv}')

if os.path.exists(seizures_csv):
    split_data(seizures_csv)
else:
    print(f'File not found: {seizures_csv}')
