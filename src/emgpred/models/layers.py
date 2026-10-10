"""Camadas customizadas do GlobalViewTransformer.

Mantidas com a mesma estrutura de pesos dos modelos treinados na tese, para que
`model.weights.h5` / `model_checkpoint.keras` antigos continuem carregando.
"""
import keras
import numpy as np
import tensorflow as tf
from keras import layers, regularizers


@keras.saving.register_keras_serializable(package="emgpred")
class LearnedPositionalEmbedding(layers.Layer):
    """Embedding posicional aprendido (estilo BERT)."""

    def __init__(self, max_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.position_embeddings = layers.Embedding(
            input_dim=max_length,
            output_dim=d_model,
            embeddings_regularizer=regularizers.l1_l2(1e-7, 1e-7),
            name="pos_embedding",
        )

    def call(self, inputs):
        x = keras.ops.cast(inputs, self.compute_dtype)
        positions = keras.ops.arange(start=0, stop=keras.ops.shape(x)[1], step=1)
        return x + keras.ops.cast(self.position_embeddings(positions), x.dtype)

    def get_config(self):
        return {**super().get_config(), "max_length": self.max_length, "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="emgpred")
class SinusoidalPositionalEncoding(layers.Layer):
    """Encoding posicional fixo (seno/cosseno)."""

    def __init__(self, max_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        pos = np.arange(max_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        pe = np.zeros((max_length, d_model))
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[tf.newaxis, : tf.shape(inputs)[1], :]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "max_length": self.max_length, "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="emgpred")
class HybridPositionalEmbedding(layers.Layer):
    """Embedding aprendido seguido do encoding sinusoidal."""

    def __init__(self, max_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.learned_embedding = LearnedPositionalEmbedding(max_length, d_model)
        self.sinusoidal_encoding = SinusoidalPositionalEncoding(max_length, d_model)

    def call(self, inputs):
        return self.sinusoidal_encoding(self.learned_embedding(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "max_length": self.max_length, "d_model": self.d_model}


POSITIONAL_EMBEDDINGS = {
    "learned": LearnedPositionalEmbedding,
    "sinu": SinusoidalPositionalEncoding,
    "hybrid": HybridPositionalEmbedding,
}


@keras.saving.register_keras_serializable(package="emgpred")
class TransformerEncoderBlock(layers.Layer):
    """Bloco encoder pre-norm (LN -> MHA -> residual, LN -> MLP -> residual)."""

    def __init__(self, n_heads: int, mlp_ratio: float, dropout: float,
                 use_causal_mask: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.use_causal_mask = use_causal_mask

    def build(self, input_shape):
        d_model = input_shape[-1]
        assert d_model % self.n_heads == 0, "d_model deve ser divisível por n_heads"
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(
            num_heads=self.n_heads, key_dim=d_model // self.n_heads, dropout=self.dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ff = keras.Sequential([
            layers.Dense(int(self.mlp_ratio * d_model), activation="gelu"),
            layers.Dropout(self.dropout),
            layers.Dense(d_model),
        ])
        super().build(input_shape)

    def call(self, x, padding_mask=None, return_attention_scores=False):
        h = self.norm1(x)
        out = self.attn(query=h, value=h, key=h, attention_mask=padding_mask,
                        use_causal_mask=self.use_causal_mask,
                        return_attention_scores=return_attention_scores)
        attn_out, attn_scores = out if return_attention_scores else (out, None)
        x = x + keras.ops.cast(attn_out, x.dtype)
        x = x + keras.ops.cast(self.ff(self.norm2(x)), x.dtype)
        return (x, attn_scores) if return_attention_scores else x

    def get_config(self):
        return {**super().get_config(), "n_heads": self.n_heads, "mlp_ratio": self.mlp_ratio,
                "dropout": self.dropout, "use_causal_mask": self.use_causal_mask}


@keras.saving.register_keras_serializable(package="emgpred")
class ClassToken(layers.Layer):
    """Prepend de um token [CLS] treinável na dimensão temporal."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model), initializer="random_normal", trainable=True, name="cls_token"
        )

    def call(self, inputs):
        cls = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        return tf.concat([cls, inputs], axis=1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}


def custom_objects():
    """Dicionário para `keras.models.load_model(..., custom_objects=custom_objects())`."""
    return {cls.__name__: cls for cls in (
        LearnedPositionalEmbedding, SinusoidalPositionalEncoding, HybridPositionalEmbedding,
        TransformerEncoderBlock, ClassToken,
    )}
