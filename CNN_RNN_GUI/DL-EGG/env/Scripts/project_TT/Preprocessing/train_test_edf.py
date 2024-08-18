import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
healthy_path = '../../../../data_EGG/healthy/'
seizures_path = '../../../../data_EGG/seizures'

# List all EDF files
healthy_files = [os.path.join(healthy_path, f) for f in os.listdir(healthy_path) if f.endswith('.edf')]
seizures_files = [os.path.join(seizures_path, f) for f in os.listdir(seizures_path) if f.endswith('.edf')]

# Split files into train and test sets (80% train, 20% test)
healthy_train, healthy_test = train_test_split(healthy_files, test_size=0.2, random_state=42)
seizures_train, seizures_test = train_test_split(seizures_files, test_size=0.2, random_state=42)

# Create directories for train and test sets
os.makedirs('data_EGG/healthy_train', exist_ok=True)
os.makedirs('data_EGG/healthy_test', exist_ok=True)
os.makedirs('data_EGG/seizures_train', exist_ok=True)
os.makedirs('data_EGG/seizures_test', exist_ok=True)

# Move files to respective directories
for file in healthy_train:
    shutil.copy(file, 'data_EGG/healthy_train/')
for file in healthy_test:
    shutil.copy(file, 'data_EGG/healthy_test/')
for file in seizures_train:
    shutil.copy(file, 'data_EGG/seizures_train/')
for file in seizures_test:
    shutil.copy(file, 'data_EGG/seizures_test/')
