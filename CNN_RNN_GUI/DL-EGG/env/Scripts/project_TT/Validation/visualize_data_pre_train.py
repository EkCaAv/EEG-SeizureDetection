import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Función para crear una carpeta si no existe
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Función para cargar y procesar los datos
def load_and_process_data(file_path, label, sample_size=None):
    df = pd.read_csv(file_path)
    df['label'] = label
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y, df

# Función para alinear los DataFrames
def align_dataframes(df1, df2):
    common_columns = df1.columns.intersection(df2.columns)
    return df1[common_columns], df2[common_columns]

# Función para graficar señales EEG
def plot_eeg_data(X, y, title, folder_path):
    plt.figure(figsize=(15, 6))
    for i in range(2):  # Plot for each class
        sample = X[y == i][0]  # Get the first sample of each class
        plt.plot(sample, label=f'Class {i}')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'{title}.png'))
    plt.close()

# Función para graficar histogramas de las características
def plot_histogram(df, title, folder_path):
    plt.figure(figsize=(15, 6))
    df.hist(bins=50, figsize=(20, 15))
    plt.suptitle(title)
    plt.savefig(os.path.join(folder_path, f'{title}.png'))
    plt.close()

# Función para graficar matriz de correlación
def plot_correlation_matrix(df, title, folder_path):
    plt.figure(figsize=(15, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.savefig(os.path.join(folder_path, f'{title}.png'))
    plt.close()

# Definir rutas de los archivos CSV
healthy_train_path = '../../../../data_EGG/healthy/cleaned_data_train.csv'
seizures_train_path = '../../../../data_EGG/seizures/cleaned_data_train.csv'
healthy_test_path = '../../../../data_EGG/healthy/cleaned_data_test.csv'
seizures_test_path = '../../../../data_EGG/seizures/cleaned_data_test.csv'

# Crear carpeta para guardar las gráficas
output_folder = 'eeg_data_visualizations'
create_folder(output_folder)

# Cargar y procesar datos
X_train_healthy, y_train_healthy, df_train_healthy = load_and_process_data(healthy_train_path, label=0)
X_train_seizures, y_train_seizures, df_train_seizures = load_and_process_data(seizures_train_path, label=1)
X_test_healthy, y_test_healthy, df_test_healthy = load_and_process_data(healthy_test_path, label=0)
X_test_seizures, y_test_seizures, df_test_seizures = load_and_process_data(seizures_test_path, label=1)

# Alinear datos de entrenamiento
X_train_healthy_df, X_train_seizures_df = align_dataframes(df_train_healthy, df_train_seizures)
X_train_healthy = X_train_healthy_df.values
X_train_seizures = X_train_seizures_df.values

# Alinear datos de prueba
X_test_healthy_df, X_test_seizures_df = align_dataframes(df_test_healthy, df_test_seizures)
X_test_healthy = X_test_healthy_df.values
X_test_seizures = X_test_seizures_df.values

# Concatenar datos de entrenamiento y prueba
X_train = np.concatenate((X_train_healthy, X_train_seizures), axis=0)
y_train = np.concatenate((y_train_healthy, y_train_seizures), axis=0)
X_test = np.concatenate((X_test_healthy, X_test_seizures), axis=0)
y_test = np.concatenate((y_test_healthy, y_test_seizures), axis=0)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Graficar y guardar las señales EEG para las clases de entrenamiento y prueba
plot_eeg_data(X_train, y_train, 'Training Data - EEG Signals', output_folder)
plot_eeg_data(X_test, y_test, 'Test Data - EEG Signals', output_folder)

# Crear DataFrames para los histogramas y matrices de correlación
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)

# Graficar y guardar histogramas de las características de entrenamiento y prueba
plot_histogram(df_train, 'Training Data - Feature Histograms', output_folder)
plot_histogram(df_test, 'Test Data - Feature Histograms', output_folder)

# Graficar y guardar matrices de correlación de las características de entrenamiento y prueba
plot_correlation_matrix(df_train, 'Training Data - Correlation Matrix', output_folder)
plot_correlation_matrix(df_test, 'Test Data - Correlation Matrix', output_folder)

print("Gráficas generadas y guardadas exitosamente en la carpeta:", output_folder)
