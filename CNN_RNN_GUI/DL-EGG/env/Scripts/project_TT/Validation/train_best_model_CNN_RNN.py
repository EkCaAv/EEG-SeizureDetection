import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras_tuner import HyperModel, RandomSearch

class CNNLSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential([
            Conv1D(filters=hp.Int('filters1', 32, 128, step=32),
                   kernel_size=hp.Choice('kernel_size1', [3, 5, 7]),
                   activation='relu', 
                   input_shape=self.input_shape,
                   padding='same',
                   kernel_regularizer=l2(hp.Float('l2_1', 1e-6, 1e-3, sampling='log'))),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout1', 0.2, 0.5, step=0.1)),

            Conv1D(filters=hp.Int('filters2', 64, 256, step=64),
                   kernel_size=hp.Choice('kernel_size2', [3, 5]),
                   activation='relu',
                   padding='same',
                   kernel_regularizer=l2(hp.Float('l2_2', 1e-6, 1e-3, sampling='log'))),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout2', 0.2, 0.5, step=0.1)),

            Conv1D(filters=hp.Int('filters3', 128, 512, step=128),
                   kernel_size=hp.Choice('kernel_size3', [3, 5]),
                   activation='relu',
                   padding='same',
                   kernel_regularizer=l2(hp.Float('l2_3', 1e-6, 1e-3, sampling='log'))),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            Dropout(rate=hp.Float('dropout3', 0.2, 0.5, step=0.1)),

            LSTM(units=hp.Int('units', 50, 150, step=50),
                 dropout=hp.Float('lstm_dropout', 0.1, 0.5, step=0.1),
                 recurrent_dropout=hp.Float('recurrent_dropout', 0.1, 0.5, step=0.1),
                 kernel_regularizer=l2(hp.Float('l2_lstm', 1e-6, 1e-3, sampling='log'))),

            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

def load_and_process_data(file_path, label, sample_size=20000):
    df = pd.read_csv(file_path)
    df['label'] = label
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def align_dataframes(df1, df2):
    common_columns = df1.columns.intersection(df2.columns)
    return df1[common_columns], df2[common_columns]

def plot_eeg_data(X, y, title):
    plt.figure(figsize=(15, 6))
    for i in range(2):  # Plot for each class
        sample = X[y == i][0]  # Get the first sample of each class
        plt.plot(sample, label=f'Class {i}')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Paths to CSV files
healthy_train_path = '../../../../data_EGG/healthy/cleaned_data_train.csv'
seizures_train_path = '../../../../data_EGG/seizures/cleaned_data_train.csv'
healthy_test_path = '../../../../data_EGG/healthy/cleaned_data_test.csv'
seizures_test_path = '../../../../data_EGG/seizures/cleaned_data_test.csv'

# Load and process data
X_train_healthy, y_train_healthy = load_and_process_data(healthy_train_path, label=0)
X_train_seizures, y_train_seizures = load_and_process_data(seizures_train_path, label=1)
X_test_healthy, y_test_healthy = load_and_process_data(healthy_test_path, label=0)
X_test_seizures, y_test_seizures = load_and_process_data(seizures_test_path, label=1)

# Align training data
X_train_healthy_df, X_train_seizures_df = align_dataframes(pd.DataFrame(X_train_healthy), pd.DataFrame(X_train_seizures))
X_train_healthy = X_train_healthy_df.values
X_train_seizures = X_train_seizures_df.values

# Align testing data
X_test_healthy_df, X_test_seizures_df = align_dataframes(pd.DataFrame(X_test_healthy), pd.DataFrame(X_test_seizures))
X_test_healthy = X_test_healthy_df.values
X_test_seizures = X_test_seizures_df.values

# Concatenate training and testing data
X_train = np.concatenate((X_train_healthy, X_train_seizures), axis=0)
y_train = np.concatenate((y_train_healthy, y_train_seizures), axis=0)
X_test = np.concatenate((X_test_healthy, X_test_seizures), axis=0)
y_test = np.concatenate((y_test_healthy, y_test_seizures), axis=0)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define input shape
input_shape = (X_train.shape[1], 1)

# Define tuner
tuner = RandomSearch(
    CNNLSTMHyperModel(input_shape),
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=1,
    directory='hyperparameter_tuning',
    project_name='CNN_LSTM_Model',
    overwrite=True
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, batch_size=16, 
             validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test set
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Save the best model
best_model.save('best_model.h5')

# Function to plot training history
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.show()

# Train the best model on the training data
history = best_model.fit(X_train, y_train, epochs=100, batch_size=8, 
                         validation_data=(X_test, y_test), 
                         callbacks=[early_stopping, reduce_lr])

# Plot training history
plot_training_history(history, 'CNN_LSTM_Model')

# Generate classification report
y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
