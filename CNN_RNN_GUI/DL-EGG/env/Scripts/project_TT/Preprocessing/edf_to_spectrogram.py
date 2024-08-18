import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal
from PIL import Image

def read_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data = raw.get_data()
    return data

def create_spectrogram(data, fs=256, nperseg=256):
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=nperseg)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Convert to dB scale for better visualization
    return Sxx_db

def save_spectrogram_image(spectrogram, filename):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', cmap='plasma', origin='lower')
    plt.colorbar(label='Intensity (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def convert_edf_to_images(edf_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path in edf_files:
        data = read_edf(file_path)
        for i, signal_data in enumerate(data):
            spectrogram = create_spectrogram(signal_data)
            image_filename = os.path.join(output_dir, f'{os.path.basename(file_path)}_channel_{i}.png')
            save_spectrogram_image(spectrogram, image_filename)

# Convert EDF files to images
healthy_files = [os.path.join('../../../../data_EGG/healthy/', f) for f in os.listdir('../../../../data_EGG/healthy/') if f.endswith('.edf')]
seizures_files = [os.path.join('../../../../data_EGG/seizures/', f) for f in os.listdir('../../../../data_EGG/seizures/') if f.endswith('.edf')]

convert_edf_to_images(healthy_files, 'data/healthy_images/')
convert_edf_to_images(seizures_files, 'data/seizures_images/')
