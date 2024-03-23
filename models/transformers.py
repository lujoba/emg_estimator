import tensorflow as tf
import complexnn
import keras
from keras import layers
from utils.utils import fft_tensor

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.linalg.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(q)[-1], tf.float32)
    logits = matmul_qk/tf.math.sqrt(d_k)
    if mask is not None:
        logits += (mask * 1e-9)
    weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.linalg.matmul(weights, v)

    return outputs, weights


class ComplexMultiHeadAttention(layers.Layer):
    def __init__(self, d_model=256, num_heads=2):
        super(ComplexMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads
        self.wq = complexnn.conv.ComplexDense(d_model)
        self.wk = complexnn.conv.ComplexDense(d_model)
        self.wv = complexnn.conv.ComplexDense(d_model)
        self.linear = complexnn.conv.ComplexDense(d_model)

    def split_head(self, x, batch_size):
        # Transpose to (batch_size, num_heads, seq_length, self.depth)
        # before return for calculation in scaled_dot_product_attention
        new_dim = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(new_dim, perm=[0, 2, 1, 3])

    def __call__(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)

        # Perform scaled dot product attention
        attn_outputs, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Retranspose to (batch_size, seq_length, num_heads, depth) to
        # reshape to original dimension
        attn_outputs = tf.transpose(attn_outputs, perm=[0, 2, 1, 3])
        outputs = tf.reshape(attn_outputs, (batch_size, -1, self.d_model))

        return outputs, attn_weights


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential([
           layers.Dense(
               dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-7) # _MklLayerNorm error
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-7) # _MklLayerNorm error

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


# Complex TransformerEncoder
class ComplexTransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        # self.attention = ComplexMultiHeadAttention(
        #     num_heads=self.num_heads, d_model=self.embed_dim
        # )
        # Complex multiplication with real and imaginary parts
        self.dense_proj = keras.Sequential([
            complexnn.conv.ComplexDense(dense_dim, activation=tf.nn.gelu),
            complexnn.conv.ComplexDense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-7) # _MklLayerNorm error
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-7) # _MklLayerNorm error

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        real_attention_output = self.attention(
            inputs[:, :, 0], inputs[:, :, 0], attention_mask=mask
        )
        imag_attention_output = self.attention(
            inputs[:, :, 1], inputs[:, :, 1], attention_mask=mask
        )
        attention_output = tf.concat([real_attention_output, imag_attention_output], axis=-1)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_input = tf.stack([real_attention_output, imag_attention_output], axis=-1)
        complex_proj_output = self.dense_proj(proj_input)

        return self.layernorm_2(proj_input + complex_proj_output)


class SignalAnalyser(tf.keras.layers.Layer):
    def __init__(
        self,
        sequence_length=100, embed_dim=256, num_channels=1,
        dense_dim=4, num_heads=1, **kwargs
        ):
        super(SignalAnalyser, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.positional_embedding = PositionalEmbedding(
            sequence_length,
            embed_dim * num_channels,
            name="frame_position_embedding"
        )
        self.transformer = TransformerEncoder(
            embed_dim * num_channels,
            dense_dim,
            num_heads,
            name="transformer_encoder"
        )
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.reshape = tf.keras.layers.Reshape((16, 16, 1))
    def call(self, inputs):
        x = layers.Reshape(
            (self.sequence_length, self.embed_dim * self.num_channels)
        )(inputs)
        x = self.positional_embedding(x)
        x = self.transformer(x)
        x = self.global_average_pooling(x)
        x = self.reshape(x)
        x = layers.Dense(128, activation=tf.keras.activations.swish())(x)
        return x

def build_model(sequence_length=100, embed_dim=256, num_channels=1, complex=False):
    dense_dim = 8
    num_heads = 2

    inputs = keras.Input(shape=(None, None))
    x = inputs

    if num_channels == 2:
        reshape = tf.keras.layers.Reshape((sequence_length, embed_dim, 1))
        x = reshape(x)
        x = tf.cast(
            x, dtype=tf.complex64, name=None
        )
        # Separate the complex and the real part in two different channels
        x = fft_tensor(x)

    x = layers.Reshape((sequence_length, embed_dim * num_channels))(x)
    x = PositionalEmbedding(
        sequence_length,
        embed_dim * num_channels,
        name="frame_position_embedding"
    )(x)
    if not complex:
        x = TransformerEncoder(
            embed_dim * num_channels,
            dense_dim,
            num_heads,
            name="transformer_layer"
        )(x)
    else: 
        x = ComplexTransformerEncoder(
            embed_dim * num_channels,
            dense_dim,
            num_heads,
            name="transformer_layer"
        )(x)
    x = layers.GlobalMaxPooling1D()(x)
    outputs = layers.Dense(embed_dim, activation="softsign")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
