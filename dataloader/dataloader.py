import os
import wfdb
import tensorflow as tf
from utils.utils import butter_lowpass_filter, normalize


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
            # Filter the high frequencies
            force_signal = butter_lowpass_filter(force_signal, cutoff, fs, order=4)
            force_signal = normalize(force_signal, -1, 1)
            # Normalize by dividing by the maximum absolute value
            # max_abs_value = tf.reduce_max(tf.abs(force_signal))
            # force_signal = tf.divide(force_signal, max_abs_value) # force_signal /= max_abs_value
            force_signal = tf.cast(force_signal, dtype=tf.float16)
            return force_signal

def create_synced_datasets(path, emg_files, force_files, window_size=100, electrodes=256):
    datasets = []
    weights_dataset = []
    for emg_file, force_file in zip(emg_files, force_files):
        emg_signal = wfdb.rdsamp(os.path.join(path, emg_file[:-4]))[0]
        emg_signal = normalize(emg_signal, -1, 1)
        emg_signal = tf.cast(emg_signal, dtype=tf.float16)

        # Pré-processing the signal
        if electrodes == 128:
            emg_signal =  electrode_selection(emg_signal)
        elif electrodes == 112:
            num_chunks = 2
            emg_signal = calculate_differential_matrices(
                electrode_selection(emg_signal), num_chunks
                )
        elif electrodes == 224:
            num_chunks = 4
            emg_signal = calculate_differential_matrices(
                emg_signal, num_chunks
            )

        force_signal = force_data(path, force_file)
        finger_dataset = synced_windows(emg_signal, force_signal, window_size=window_size)
        datasets.append(finger_dataset)

    dataset = datasets[0]
    for i in range(1, len(datasets)):
        dataset = dataset.concatenate(datasets[i])
    return dataset

# @tf.function
def create_synced_datasets_generator(paths, emg_files_list, force_files_list, window_size=100, electrodes=256):
    for path in paths:
        for emg_file, force_file in zip(emg_files_list, force_files_list):
            yield from create_synced_datasets(path, emg_file, force_file, window_size=window_size, electrodes=electrodes)

def calculate_differential_matrices(input_array, num_chunks):
    """
    Calculates differential matrices for each sample in a 1D TensorFlow tensor.
    Args:
        input_array: A 2D TensorFlow tensor with shape [num_samples, 256].
        num_chunks: The number of chunks to divide each sample into.
    Returns:
        A 3D TensorFlow tensor with shape [num_samples, num_chunks, 7].
        Contains differential matrices for each sample and chunk.
    Raises:
        ValueError: If the sample length (256) is not divisible by num_chunks * 8.
    """
    
    # Check if sample length is divisible by required amount
    if 256 % (num_chunks * 8) != 0:
        raise ValueError("Sample length must be divisible by num_chunks * 8")
    # Iterate over each sample in the input tensor
    differential_matrices = []
    for sample in input_array:
        # Split the sample into chunks of appropriate size
        chunk_length = len(sample) // num_chunks
        chunks = tf.split(sample, num_or_size_splits=len(sample) // chunk_length)
        # Reshape each chunk into a matrix of size 8x8
        matrices = [tf.reshape(chunk, (8, 8)) for chunk in chunks]
        # Apply the differential matrix operation to each matrix
        sample_diff_matrices = []
        for matrix in matrices:
            differential_matrix = matrix[1:, :] - matrix[:-1, :]
            sample_diff_matrices.append(differential_matrix)
        # Concatenate the differential matrices for this sample
        sample_result = tf.concat(sample_diff_matrices, axis=0)
        # Add the sample's differential matrices to the result list
        differential_matrices.append(sample_result)
    
    # Combine differential matrices for all samples into a single tensor
    result = tf.stack(differential_matrices, axis=0)
    # Reshape the combined tensor into a 2D array
    return tf.reshape(result, (tf.shape(result)[0], -1))

def electrode_selection(emg_signal):
    """
    Selects a subset of electrodes from an EMG signal.
    Args:
        emg_signal: A 2D TensorFlow tensor representing the EMG signal, with
        shape (num_electrodes, time_steps).
    Returns:
        A 2D TensorFlow tensor representing the selected EMG signal, with shape
        (64, time_steps), containing channels from electrodes 64 to 127 and
        192 to 255 (inclusive).
    Raises:
        ValueError: If the input `emg_signal` does not have a rank of 2.
    """
    
    # Check if input has the expected rank
    if len(emg_signal.shape) != 2:
        raise ValueError("Input emg_signal must have a rank of 2.")
    # Create a tensor of indices for the desired electrodes
    indices = tf.concat([tf.range(64, 128), tf.range(192, 256)], axis=0)
    # Select the desired electrodes using tf.gather
    emg_signal = tf.gather(emg_signal, indices, axis=1)
    # Cast the selected signal to float16 for consistency
    emg_signal = tf.cast(emg_signal, dtype=tf.float16)
    return emg_signal
