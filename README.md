# EEG-SeizureDetection

### README (English Version)

# EEG Seizure Detection Project

## Overview

This project focuses on detecting abnormal electrical impulses in the brain using Convolutional Neural Networks (CNN) to identify seizures and epilepsy episodes. The data was sourced from two public datasets, [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) and [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).

### Key Results

The most recent model achieved a **precision of 0.78** in detecting seizures, demonstrating strong performance in identifying seizure episodes. However, it is crucial to highlight that while this model excels in detecting seizures, its balance between classes can be further improved. Another model, with a more balanced approach, achieved an accuracy of 0.54 and a precision of 0.54. This comparative analysis highlights ongoing efforts to optimize the model for both seizure detection accuracy and class balance.

### Model Training and Development

Initially, EEG data in EDF format was converted to CSV files, but this approach led to overfitting due to the high dimensionality of the data. The project then transitioned to converting EDF files into spectrogram images, which provided a more robust input for the CNN models, resulting in improved detection performance.

The project involved experimenting with various models and hyperparameters to achieve the best results. The current model continues to undergo training and optimization to enhance its accuracy and overall performance.

### Repository

The full project, including all models and training scripts, can be found in this repository: [EEG-SeizureDetection](https://github.com/EkCaAv/EEG-SeizureDetection/blob/main/CNN_RNN_GUI/DL-EGG/env/Scripts/project_TT/classification_report_v1.txt).

Author

Erika Isabel Caita Ávila

---

### README (Spanish Version)

# Proyecto de Detección de Convulsiones en EEG

## Descripción General

Este proyecto se centra en la detección de impulsos eléctricos anormales en el cerebro utilizando Redes Neuronales Convolucionales (CNN) para identificar episodios de convulsiones y epilepsia. Los datos provienen de dos conjuntos de datos públicos, [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) y [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).

### Resultados Clave

El modelo más reciente logró una **precisión de 0.78** en la detección de convulsiones, demostrando un desempeño sólido en la identificación de episodios convulsivos. Sin embargo, es importante destacar que, si bien este modelo sobresale en la detección de convulsiones, su equilibrio entre clases puede mejorarse. Otro modelo, con un enfoque más equilibrado, alcanzó una precisión de 0.54 y un accuracy de 0.54. Este análisis comparativo resalta los esfuerzos en curso para optimizar el modelo tanto en precisión de detección de convulsiones como en el balance de clases.

### Entrenamiento y Desarrollo del Modelo

Inicialmente, los datos de EEG en formato EDF se convirtieron a archivos CSV, pero este enfoque llevó a un sobreajuste debido a la alta dimensionalidad de los datos. El proyecto luego pasó a convertir los archivos EDF en imágenes de espectrogramas, lo que proporcionó una entrada más robusta para los modelos CNN, resultando en un mejor rendimiento de detección.

El proyecto implicó la experimentación con varios modelos e hiperparámetros para lograr los mejores resultados. El modelo actual continúa en proceso de entrenamiento y optimización para mejorar su precisión y rendimiento general.

### Repositorio

El proyecto completo, incluidos todos los modelos y scripts de entrenamiento, se puede encontrar en este repositorio: [EEG-SeizureDetection](https://github.com/EkCaAv/EEG-SeizureDetection/blob/main/CNN_RNN_GUI/DL-EGG/env/Scripts/project_TT/classification_report_v1.txt).

Autora

Erika Isabel Caita Ávila
