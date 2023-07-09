# Important functions
#todo convert to tf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.signal import correlate2d
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


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
    