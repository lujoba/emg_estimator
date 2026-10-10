"""Baselines do artigo (128 eletrodos), iguais aos do notebook `PhD_Baselines`
(gerado por `cache_saliency/make_baselines.py`).

| nome          | arquitetura                                                      | parâmetros |
|---------------|------------------------------------------------------------------|-----------:|
| `conv2plus1d` | Conv(2+1)D residual com BatchNormalization (run `resnet18`)      |    245.237 |
| `resnet50`    | keras ResNet50 sem pré-treino, canal repetido 3× (run `resnet50`) | 23.597.957 |
| `pet`         | External Attention em paralelo com o MLP (run `paralel_trans`)   |    215.045 |
| `raulnet`     | CNN 3D + MLP com SMU, 2 grades de 64 (run `raul`); fora do artigo | 22.710.906 |

Todos recebem a janela como (tempo, eletrodos, 1), igual ao GVT.
"""
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from emgpred.constants import FS_EMG

EXPECTED_PARAMS = {"conv2plus1d": 245_237, "resnet50": 23_597_957, "pet": 215_045, "raulnet": 22_710_906}


# ---------------------------------------------------------------------------
# Conv(2+1)D (run `resnet18`): blocos residuais com convolução espacial (1×k) e temporal (k×1)
# ---------------------------------------------------------------------------
def _conv2plus1d(filters, kernel_size, padding="same"):
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size=(1, kernel_size[1]), padding=padding),
        layers.Conv2D(filters, kernel_size=(kernel_size[0], 1), padding=padding),
    ])


def _residual_block(x, filters, kernel_size):
    # BatchNormalization como no run `resnet18` (o tutorial de vídeo do TF usa LayerNormalization)
    out = _conv2plus1d(filters, kernel_size)(x)
    out = layers.ReLU()(layers.BatchNormalization()(out))
    out = layers.BatchNormalization()(_conv2plus1d(filters, kernel_size)(out))
    res = x
    if out.shape[-1] != x.shape[-1]:
        res = layers.BatchNormalization()(layers.Dense(out.shape[-1])(x))
    return layers.add([res, out])


def build_conv2plus1d(time_steps=100, n_channels=128, output_dim=5, widths=(16, 32, 64, 128)):
    inputs = keras.Input(shape=(time_steps, n_channels, 1), name="emg")
    x = layers.ReLU()(layers.BatchNormalization()(_conv2plus1d(widths[0], (3, 7))(inputs)))
    for i, w in enumerate(widths, start=1):
        x = layers.Resizing(-(time_steps // -(2 ** i)), -(n_channels // -(2 ** i)))(x)
        x = _residual_block(x, w, (3, 3))
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, layers.Dense(output_dim)(x), name="Conv2Plus1D")


# ---------------------------------------------------------------------------
# ResNet-50 (run `resnet50`): imagem T×S, canal repetido 3×, sem pesos pré-treinados
# ---------------------------------------------------------------------------
def build_resnet50(time_steps=100, n_channels=128, output_dim=5):
    inputs = keras.Input(shape=(time_steps, n_channels, 1), name="emg")
    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    backbone = keras.applications.ResNet50(include_top=False, weights=None, pooling="avg",
                                           input_shape=(time_steps, n_channels, 3))
    return keras.Model(inputs, layers.Dense(output_dim)(backbone(x)), name="ResNet50")


# ---------------------------------------------------------------------------
# PET (run `paralel_trans`): External Attention em paralelo com o MLP
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="emgpred")
class ExternalAttention(layers.Layer):
    """Atenção com memória externa de S tokens aprendíveis (complexidade O(n))."""

    def __init__(self, d_model, S=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.S = d_model, S
        self.M = self.add_weight(shape=(S, d_model), initializer="random_normal", trainable=True,
                                 name="external_memory")
        self.W_k = layers.Dense(S, use_bias=False)
        self.W_v = layers.Dense(d_model, use_bias=False)

    def call(self, x):
        attn = tf.nn.softmax(self.W_k(x), axis=-1)   # (B, L, S)
        return self.W_v(tf.matmul(attn, self.M))      # (B, L, d_model)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "S": self.S}


@keras.saving.register_keras_serializable(package="emgpred")
class PETEncoderLayer(layers.Layer):
    def __init__(self, d_model, dff, S=64, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.dff, self.S, self.rate = d_model, dff, S, rate
        self.ext_attn = ExternalAttention(d_model, S)
        self.ffn = keras.Sequential([layers.Dense(dff, activation="relu"), layers.Dense(d_model)])
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=None):
        attn = self.dropout1(self.ext_attn(x), training=training)
        ffn = self.dropout2(self.ffn(x), training=training)
        return self.layernorm(x + attn + ffn)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "dff": self.dff, "S": self.S, "rate": self.rate}


def build_pet(time_steps=100, n_channels=128, output_dim=5, num_layers=2, d_model=128, dff=256, S=64, rate=0.1):
    inputs = keras.Input(shape=(time_steps, n_channels, 1), name="emg")
    x = layers.Reshape((time_steps, n_channels))(inputs)
    x = layers.Dense(d_model, activation="relu")(x)
    for _ in range(num_layers):
        x = PETEncoderLayer(d_model, dff, S, rate)(x)
    x = layers.GlobalMaxPooling1D()(x)
    return keras.Model(inputs, layers.Dense(output_dim)(x), name="PETTransformer")


# ---------------------------------------------------------------------------
# RaulNet (run `raul`): CNN 3D + MLP com SMU, eletrodos em 2 grades de 64. Fora do artigo (~2 h/época).
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="emgpred")
class SMU(layers.Layer):
    """f(x) = x * sigmoid(beta * x), beta treinável."""

    def build(self, input_shape):
        self.beta = self.add_weight(shape=(1,), initializer=keras.initializers.Constant(0.25), name="beta")

    def call(self, x):
        return x * tf.nn.sigmoid(self.beta * x)


@keras.saving.register_keras_serializable(package="emgpred")
class CircularPadding3D(layers.Layer):
    """Padding circular no eixo das grades, para o kernel enxergar a grade vizinha nas bordas."""

    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, x):
        return tf.concat([x[:, -self.padding:], x, x[:, :self.padding]], axis=1)

    def get_config(self):
        return {**super().get_config(), "padding": self.padding}


@keras.saving.register_keras_serializable(package="emgpred")
class CausalLowPass(layers.Layer):
    """Passa-baixa causal de 1ª ordem (y[t] = a·y[t-1] + (1-a)·x[t]) ao longo do tempo, dentro da janela."""

    def __init__(self, fc=20.0, fs=FS_EMG, time_steps=100, **kwargs):
        super().__init__(**kwargs)
        self.fc, self.fs, self.time_steps = fc, fs, time_steps
        a = np.exp(-2 * np.pi * fc / fs)
        t = np.arange(time_steps)
        lag = t[:, None] - t[None, :]
        self.kernel = tf.constant(np.where(lag >= 0, (1 - a) * a ** np.maximum(lag, 0), 0.0), tf.float32)

    def call(self, x):   # x: (B, T, C, 1)
        return tf.einsum("tk,bkcf->btcf", self.kernel, x)

    def get_config(self):
        return {**super().get_config(), "fc": self.fc, "fs": self.fs, "time_steps": self.time_steps}


def build_raulnet(time_steps=100, n_channels=128, output_dim=5, n_grids=2, grid_size=64, second_channel="passa_baixa"):
    """second_channel: 'passa_baixa' (EMG filtrado a 20 Hz, causal, dentro da janela) ou 'copia' (EMG repetido)."""
    inputs = keras.Input(shape=(time_steps, n_channels, 1), name="emg")
    second = CausalLowPass(time_steps=time_steps)(inputs) if second_channel == "passa_baixa" else inputs
    x = layers.Concatenate(axis=-1)([inputs, second])          # (T, E, 2)
    x = layers.Permute((2, 1, 3))(x)                            # (E, T, 2)
    x = layers.Reshape((n_grids, grid_size, time_steps, 2))(x)  # (grades, eletrodos da grade, T, 2)

    x = layers.Conv3D(128, kernel_size=(1, 1, 31), strides=(1, 1, 8), padding="valid")(x)
    x = SMU()(layers.BatchNormalization()(x))
    x = layers.Dropout(0.25)(x)

    x = CircularPadding3D(padding=2)(x)
    x = layers.Conv3D(16, kernel_size=(5, 32, 18), dilation_rate=(1, 2, 1), padding="same")(x)
    x = SMU()(layers.BatchNormalization()(x))

    x = layers.Conv3D(32, kernel_size=(5, 9, 1), padding="valid")(x)
    x = SMU()(layers.BatchNormalization()(x))

    x = layers.Dropout(0.4)(layers.Flatten()(x))
    x = SMU()(layers.Dense(512)(x))
    x = SMU()(layers.Dense(512)(x))
    return keras.Model(inputs, layers.Dense(output_dim)(x), name="RaulNet")


MODELS = {"conv2plus1d": build_conv2plus1d, "resnet50": build_resnet50, "pet": build_pet, "raulnet": build_raulnet}
