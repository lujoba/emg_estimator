import os
import wfdb
import tensorflow as tf
from scipy import interpolate
from utils.utils import *


def force_data(path, force_file, window_size, output_shape=5):
    """
    Loads and preprocesses a force signal from a file.
    
    Arguments:
        - `path`: Path to the directory containing the force data file.
        - `force_file`: Filename of the force data file.
    
    Returns:
        A numpy array containing the filtered force signal.
    
    Raises:
        ValueError: If the finger name is not found in the filename.
        
    Details:
        - This function assumes the force data file follows a naming convention where the finger name is part of the filename.
        - It extracts the force signal for the specified finger from the file.
        - The first 5 samples are discarded, potentially corresponding to non-data values.
        - The signal is then filtered using a 4th order low-pass Butterworth filter with a specified cutoff frequency.
    
    Example usage:
    
    ```python
    force_signal = force_data(path, "finger2_data.txt")
    ```
    """
    finger_names = ["finger1", "finger2", "finger3", "finger4", "finger5"]
    # Sample rate and desired cutoff frequency (in Hz)
    fs = 100.0
    cutoff = 5.0
    for i, name in enumerate(finger_names):
        if name in force_file and i <= output_shape:
            force_signal = wfdb.rdsamp(os.path.join(path, force_file[:-4]))[0]
            force_signal = force_signal[int(window_size//20):]
            # Filter the high frequencies
            force_signal = butter_lowpass_filter(force_signal, cutoff, fs, order=4)
            # Normalize by dividing by the maximum absolute value
            # max_abs_value = tf.reduce_max(tf.abs(force_signal))
            force_signal = tf.divide(force_signal, 0.3)
            force_signal = tf.cast(force_signal, dtype=tf.float16)
            return force_signal

def interpolate_force_to_emg(force_signal, emg_signal):
    """
    Interpolates a force signal to have the same shape as an EMG signal.

    Parameters:
        - force_signal: numpy array with the input force signal, with shape (n_samples_force, n_channels)
        - emg_signal: numpy array with the reference EMG signal, with shape (n_samples_emg, n_channels)
    
    Returns:
        - interpolated_signals: numpy array with the interpolated force signals, with shape (n_samples_emg, n_channels)
    
    Example usage:
        - interpolated_force = interpolate_force_to_emg(force_signal, emg_signal)
    """
    interpolated_signals = np.zeros((emg_signal.shape[0], force_signal.shape[1]))  # Array to store interpolated signals

    for i in range(force_signal.shape[1]):  # For each force channel
        x = np.arange(0, force_signal.shape[0])  # Original indices of the force signal
        y = force_signal[:, i]  # Value of the force signal
        f = interpolate.interp1d(x, y, kind='cubic')  # Cubic interpolation function
        x_new = np.linspace(0, force_signal.shape[0] - 1, emg_signal.shape[0])  # New indices for the interpolated signal
        interpolated_signals[:, i] = f(x_new)  # Interpolate the force signal to the length of the EMG signal

    return interpolated_signals

def synced_windows(emg_signal, force_signal, window_size=100, shift=1):
    """
    Create windows from EMG and force signals using tf.slice.

    Arguments:
        emg_signal: EMG signal tensor.
        force_signal: Force signal tensor.
        emg_window_size: Size of the EMG window (default: 100).
        force_window_size: Size of the force window (default: 1).
    
    Returns:
        A TensorFlow dataset containing synchronized EMG and force windows.
    """
    # Adjust the signal
    emg_windows = tf.data.Dataset.from_tensor_slices(emg_signal)
    force_windows = tf.data.Dataset.from_tensor_slices(force_signal)
    
    # Extract EMG windows
    emg_windows = emg_windows.window(window_size, shift=shift, drop_remainder=True)
    force_windows = force_windows.window(1, shift=shift)

    # Convert to windows of desired size
    emg_windows = emg_windows.flat_map(lambda x: x.batch(window_size))
    force_windows = force_windows.flat_map(lambda x: x.batch(1))

    # Create the dataset
    dataset = tf.data.Dataset.zip((emg_windows, force_windows))
    return dataset

def create_synced_datasets(path, emg_files, force_files, window_size=100,  electrodes=256, shift=1):
    """
    Creates a TensorFlow dataset by loading, processing, and combining synchronized EMG and force signals from multiple files.
    
    Arguments:
        - `path`: Path to the directory containing the data files.
        - `emg_files`: List of filenames for the EMG files.
        - `force_files`: List of filenames for the force files.
        - `window_size`: Size of the window in samples to split the signals (default: 100).
        - `stride`: Step size of the window in samples when splitting the signals (default: 1).
    
    Returns:
        A TensorFlow dataset containing synchronized windows of EMG and force signals from all the files.
    """
    datasets = []
    for emg_file, force_file in zip(emg_files, force_files):
        # Load EMG and force signals
        emg_signal = wfdb.rdsamp(os.path.join(path, emg_file[:-4]))[0]
        max_abs_value = tf.reduce_max(tf.abs(emg_signal))
        emg_signal = tf.divide(emg_signal[:-window_size], max_abs_value)
        emg_signal = tf.cast(emg_signal, dtype=tf.float32)

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

        force_signal = force_data(path, force_file, window_size)
        force_signal = interpolate_force_to_emg(force_signal, emg_signal)

        # Create the dataset for the finguers
        finger_dataset = synced_windows(emg_signal, force_signal, window_size=window_size, shift=shift)
        # Apply preprocessing to force signals
        finger_dataset = finger_dataset.map(lambda x, y: (x, preprocess_force(y)))

        datasets.append(finger_dataset)

    dataset = datasets[0]
    for i in range(1, len(datasets)):
        dataset = dataset.concatenate(datasets[i])
    return dataset

def create_synced_datasets_generator(paths, emg_files_list, force_files_list, window_size=100, electrodes=256, shift=1):
    """
    Yields datasets generated from processing multiple data files in a generator fashion.

    Arguments:
        - `paths`: A list of paths to the directories containing the data files.
        - `emg_files_list`: A list of lists containing filenames for the EMG files for each path.
        - `force_files_list`: A list of lists containing filenames for the force files for each path.
        - `window_size`: Size of the window in samples to split the signals (default: 100).

    Yields:
        A TensorFlow dataset containing synchronized windows of EMG and force signals from a single file pair.
    """
    for path in paths:
        for emg_file, force_file in zip(emg_files_list, force_files_list):
            yield from create_synced_datasets(path, emg_file, force_file, window_size=window_size, electrodes=electrodes, shift=shift)

def preprocess_force(force):
    """
    Preprocesses a force tensor by removing the first dimension.

    Args:
        force: A TensorFlow tensor representing force data.

    Returns:
        The preprocessed force tensor with the first dimension removed.
    """
    force = tf.squeeze(force, axis=0)
    return force

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
