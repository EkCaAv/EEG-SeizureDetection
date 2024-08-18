import os
import mne
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras_tuner import HyperModel, RandomSearch

def read_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data = raw.get_data()
    return data.T  # Transponer para que las dimensiones sean (n_samples, n_channels)

# Supongamos que tienes etiquetas para tus datos
def load_and_process_edf(files, label):
    data_list = []
    labels_list = []
    for file in files:
        data = read_edf(file)
        data_list.append(data)
        labels = np.full(data.shape[0], label)
        labels_list.append(labels)
    return np.vstack(data_list), np.hstack(labels_list)

# Paths to train and test directories
healthy_train_path = 'data_EGG/healthy_train/'
seizures_train_path = 'data_EGG/seizures_train/'
healthy_test_path = 'data_EGG/healthy_test/'
seizures_test_path = 'data_EGG/seizures_test/'

# List all EDF files in train and test directories
healthy_train_files = [os.path.join(healthy_train_path, f) for f in os.listdir(healthy_train_path) if f.endswith('.edf')]
seizures_train_files = [os.path.join(seizures_train_path, f) for f in os.listdir(seizures_train_path) if f.endswith('.edf')]
healthy_test_files = [os.path.join(healthy_test_path, f) for f in os.listdir(healthy_test_path) if f.endswith('.edf')]
seizures_test_files = [os.path.join(seizures_test_path, f) for f in os.listdir(seizures_test_path) if f.endswith('.edf')]

# Load and process data
X_train_healthy, y_train_healthy = load_and_process_edf(healthy_train_files, label=0)
X_train_seizures, y_train_seizures = load_and_process_edf(seizures_train_files, label=1)
X_test_healthy, y_test_healthy = load_and_process_edf(healthy_test_files, label=0)
X_test_seizures, y_test_seizures = load_and_process_edf(seizures_test_files, label=1)

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

class CNNLSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # Añadir regularización L2
            Dropout(0.5),  # Añadir dropout
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Añadir regularización L2
            Dropout(0.5),  # Añadir dropout
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # Añadir regularización L2
            Dropout(0.5),  # Añadir dropout
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

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
best_model.save('best_cnn_model.h5')

# Train the best model on the training data
history = best_model.fit(X_train, y_train, epochs=100, batch_size=8, 
                         validation_data=(X_test, y_test), 
                         callbacks=[early_stopping, reduce_lr])

# Plot training history
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
