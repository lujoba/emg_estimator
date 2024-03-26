# Important functions
#todo convert to tf
import numpy as np
import yaml
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.signal import correlate2d
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


# TODO what I really use 
def normalize(tensor, target_min, target_max):
    tensor = tf.cast(tensor, tf.float32)
    target_min = tf.cast(target_min, tf.float32)
    target_max = tf.cast(target_max, tf.float32)
    return tf.add(
        tf.divide(
            tf.multiply(
                tf.subtract(tensor, tf.reduce_min(tensor)),
                tf.subtract(target_max, target_min)
                ),
            tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor))
        ),
        target_min
    )

def calc_mse(tensor1, tensor2):
    mse = tf.keras.metrics.mean_squared_error(tensor1, tensor2)
    return tf.reduce_mean(mse).numpy()

def apply_filter(
    data, fs=2048, lowcut=350, highcut=1650, order=8, freq=50, Q=30
):
    # Define the Nyquist frequency
    nyq = 0.5 * fs

    # Define the filter parameters
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Apply the bandpass filter
    data_bp = filtfilt(b, a, data)

    # Define the notch filter parameters
    notch_freq = freq / nyq
    notch_bandwidth = notch_freq / Q

    # Apply the notch filter
    b_notch, a_notch = iirnotch(notch_freq, notch_bandwidth)
    data_filtered = filtfilt(b_notch, a_notch, data_bp)

    return data_filtered

def r2(predicted, true):
    predicted = tf.cast(predicted, dtype=tf.float32)
    true = tf.reshape(
        tf.cast(
            true, dtype=tf.float32
        )
        , shape=(-1,)
    )
    correlation = tfp.stats.correlation(predicted, true, event_axis=None)
    r_squared = tf.math.pow(correlation, 2)
    return r_squared

def estimated_autocorrelation(x):
    n = len(x)
    x = x - tf.reduce_mean(x)
    r = correlate2d(x, x, mode = 'same')[-n:]
    # assert np.allclose(
    #     r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)])
    #     )
    result = tf.divide(
        r,
        tf.transpose([tf.multiply(
            tf.math.reduce_variance(x),
            tf.cast(tf.range(n, 0, -1), tf.float32)
        )])
    )
    return result

def r2(predicted, true):
    return r2_score(predicted, true)

def make_plot(prediction_data, real_data, title):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(16, 8))
    mse_sum = calc_mse(prediction_data, real_data[5:])
    title = f"{title} {mse_sum:.4f}"
    plt.title(title)
    for i in range(5):
        mse = calc_mse(prediction_data[:,i], real_data[:,i][5:])
        ax[i].plot(real_data[:,i][5:], label='Original Force')
        ax[i].plot(prediction_data[:,i][5:], label='Predicted Force')
        ax[i].text(
            0, 1.15, f"Validation MSE: {mse:.4f}",transform=ax[i].transAxes,
            ha='center', va='center', fontsize=12
        )
        ax[i].set_title(f"Finger {i+1}")
        ax[i].set_xlabel("Samples")
        ax[i].set_ylabel("Values")
    plt.tight_layout()
    plt.legend()
    plt.show()

#--- Novas Funções ---#
# Training
def load_config(config_file: str) -> dict:
    """
    Loads a configuration dictionary from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary.

    Raises:
        IOError: If the file cannot be opened or read.
        YAMLError: If the YAML file is invalid.
    """

    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}") from exc
    return config

def fft_tensor(input_tensor):
    # Assume que input_tensor tem o formato [batch_size, height, width, num_channels]

    # Aplicar a FFT ao longo dos eixos height e width (últimos dois eixos)
    fft_result = tf.signal.fft2d(input_tensor)

    # Dividir o resultado da FFT em parte real e imaginária
    real_part = tf.math.real(fft_result)
    imag_part = tf.math.imag(fft_result)

    # Criar um novo tensor combinando as partes real e imaginária
    complex_tensor = tf.stack([real_part, imag_part], axis=-1)

    return complex_tensor

def sse(y_true, y_pred):
    return tf.reduce_sum(
        tf.square(y_true - y_pred)
    )

def loss(signal, estimate):
    # Calcula a potência do sinal
    power = tf.reduce_sum(tf.square(100 * signal), axis=1)

    # Pondera a função de perda de acordo com a potência
    weighted_loss = power * tf.keras.losses.mse(estimate, signal)

    return weighted_loss

# Validation
def mse_per_channel_slice(y_true, y_pred, slice_size):
    """
    Mean squared error (MSE) per channel for a specified slice.

    Args:
        y_true: True labels (shape: [batch_size, num_channels]).
        y_pred: Predicted labels (shape: [batch_size, num_channels]).
        slice_size: Size of the slice for each channel.

    Returns:
        MSE per channel for the specified slice (shape: [num_channels]).
    """
    # Slice the true and predicted labels for each channel
    y_true_slices = [y_true[:, i][slice_size*i:slice_size*(i+1)] for i in range(y_true.shape[1])]
    y_pred_slices = [y_pred[:, i][slice_size*i:slice_size*(i+1)] for i in range(y_pred.shape[1])]

    # Calculate MSE for each channel slice
    mse_per_channel = np.array([np.mean((true - pred)**2) for true, pred in zip(y_true_slices, y_pred_slices)])
    return mse_per_channel

def load_model_from_checkpoint(model, model_name, subject_number, DA):
    # Define the directory path for the checkpoint
    if not DA:
        checkpoint_dir = f"../logs/models/{model_name}/na_all_fingers/subject_{subject_number}/"
    else:
        checkpoint_dir = f"../logs/models/{model_name}/wa_all_fingers/subject_{subject_number}/"

    # Create a tf.train.Checkpoint object with the model
    checkpoint = tf.train.Checkpoint(model=model)

    # Restore the saved values in the model from the checkpoint files
    status = checkpoint.restore(checkpoint_dir).expect_partial()

    # Check if the restoration was successful
    if tf.train.latest_checkpoint(checkpoint_dir) is None:
        print(f"Checkpoint for subject {subject_number} not found or failed to restore.")
    else:
        print(f"Checkpoint for subject {subject_number} restored successfully.")

    return status

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def standardize(input_tensor, axes=None, epsilon=1e-8):
    """
    Standardizes the input tensor along the specified axes.
    
    Args:
    - input_tensor: Input tensor to be standardized.
    - axes: Axes along which to compute mean and variance. If None, computes across all axes.
    - epsilon: Small value to prevent division by zero.
    
    Returns:
    - Standardized tensor.
    """
    mean, variance = tf.nn.moments(input_tensor, axes=axes)
    standardized_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)
    return standardized_tensor