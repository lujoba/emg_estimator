import os
import wfdb
import tensorflow as tf
from utils.utils import normalize
from scipy import signal

from scipy.signal import butter, filtfilt, iirnotch

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def synced_windows(emg_signal, force_signal, window_size=100):
    emg_samples_per_force_sample = 2048 // 100
    num_force_samples = force_signal.shape[0] - window_size
    # Calcula o número de janelas de EMG a serem geradas
    num_emg_windows = (
        emg_signal.shape[0] - window_size*emg_samples_per_force_sample
    ) // emg_samples_per_force_sample
    # Inicializa os arrays para armazenar as janelas de EMG e força
    emg_windows = tf.zeros(
        (
            num_force_samples, window_size, emg_signal.shape[1],
        )
    ).numpy()
    force_windows = tf.zeros(
        (num_force_samples, force_signal.shape[-1])
    ).numpy()
    # Amostra o sinal de EMG de acordo com a relação de frequência
    emg_step = emg_samples_per_force_sample
    for i in range(num_force_samples):
        start = i*emg_step
        end = start + window_size
        try:
            emg_windows[i] = emg_signal[start:end]
            force_windows[i] = force_signal[i + window_size // 2]
        except ValueError:
            pass
        # Atualiza o passo do EMG de acordo com a relação de frequência
        if i >= 2:
            emg_step = round(20.48)
            if i % 2 == 0:
                emg_step += 1
    dataset = tf.data.Dataset.from_tensor_slices((emg_windows, force_windows))
    return dataset

def force_data(path, force_file):
    finger_names = ["finger1", "finger2", "finger3", "finger4", "finger5"]
    # Sample rate and desired cutoff frequency (in Hz)
    fs = 100.0
    cutoff = 5.0
    for i, name in enumerate(finger_names):
        if name in force_file:
            force_signal = wfdb.rdsamp(os.path.join(path, force_file[:-4]))[0]
            force_signal = force_signal[5:]
            ######################################################
            # Filter the high frequencies
            ######################################################
            force_signal = butter_lowpass_filter(force_signal, cutoff, fs, order=4)
            return force_signal

def create_synced_datasets(path, emg_files, force_files, window_size=100):
    datasets = []
    weights_dataset = []
    for emg_file, force_file in zip(emg_files, force_files):
        emg_signal = wfdb.rdsamp(os.path.join(path, emg_file[:-4]))[0]
        emg_signal = tf.cast(emg_signal, dtype=tf.float32)
        force_signal = force_data(path, force_file)
        finger_dataset = synced_windows(emg_signal, force_signal)
        datasets.append(finger_dataset)

    dataset = datasets[0]
    for i in range(1, len(datasets)):
        dataset = dataset.concatenate(datasets[i])
    return dataset

# @tf.function
def create_synced_datasets_generator(paths, emg_files_list, force_files_list, window_size=100):
    for path in paths:
        for emg_file, force_file in zip(emg_files_list, force_files_list):
            yield from create_synced_datasets(path, emg_file, force_file, window_size=window_size)
