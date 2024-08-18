import os
import pandas as pd

def load_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    # Previsualizar datos
    print("Primeras filas del dataframe:")
    print(df.head())

    # Información del dataframe
    print("\nInformación del dataframe:")
    print(df.info())

    # Resumen estadístico
    print("\nResumen estadístico:")
    print(df.describe())

    # Identificar valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # Rellenar valores nulos con la media de la columna
    df.fillna(df.mean(), inplace=True)

    # Verificar nuevamente valores nulos
    print("\nValores nulos después de la imputación:")
    print(df.isnull().sum())

    return df

def process_folder(folder_path):
    df = load_data(folder_path)
    df_cleaned = preprocess_data(df)
    # Guardar datos limpios
    cleaned_csv_path = os.path.join(folder_path, 'cleaned_data.csv')
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    print(f"Datos limpios guardados en: {cleaned_csv_path}")

if __name__ == "__main__":
    base_path = '../../../../data_EGG'
    folder = 'seizures'
    folder_path = os.path.join(base_path, folder)
    process_folder(folder_path)
