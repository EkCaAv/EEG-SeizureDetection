import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def convert_edf_to_images(edf_files, output_folder):
    for edf_file in edf_files:
        # Load EDF file
        raw = mne.io.read_raw_edf(edf_file, preload=True)

        # Filter only EEG channels and apply montage
        eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw_eeg = raw.copy().pick_channels([raw.ch_names[i] for i in eeg_channels])
        if len(raw_eeg.ch_names) > 0:
            raw_eeg.set_montage('standard_1020')

        # Create spectrograms for all channels
        data, times = raw[:, :]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, channel in enumerate(raw.ch_names):
            plt.specgram(data[i], Fs=raw.info['sfreq'], NFFT=256, noverlap=128, cmap='jet')
            plt.title(f'Spectrogram - {channel}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.savefig(os.path.join(output_folder, f'{os.path.basename(edf_file).replace(".edf", "")}_{channel}.png'))
            plt.close(fig)

# Paths to EDF files
healthy_files = [os.path.join('../../../../data_EGG/healthy/', f) for f in os.listdir('../../../../data_EGG/healthy/') if f.endswith('.edf')]
seizures_files = [os.path.join('../../../../data_EGG/seizures/', f) for f in os.listdir('../../../../data_EGG/seizures/') if f.endswith('.edf')]

# Convert EDF files to images
convert_edf_to_images(healthy_files, 'data/healthy_images/EGG/')
convert_edf_to_images(seizures_files, 'data/seizures_images/EGG/')
