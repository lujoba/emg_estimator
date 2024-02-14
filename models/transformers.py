import tensorflow as tf
import keras
from keras import layers

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
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


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

def build_model(sequence_length=100, embed_dim=256, num_channels=1):
    dense_dim = 8
    num_heads = 2

    inputs = keras.Input(shape=(None, None))
    x = layers.Reshape((sequence_length, embed_dim * num_channels))(inputs)
    x = PositionalEmbedding(
        sequence_length,
        embed_dim * num_channels,
        name="frame_position_embedding"
    )(x)
    x = TransformerEncoder(
        embed_dim * num_channels,
        dense_dim,
        num_heads,
        name="transformer_layer"
    )(x)
    x = layers.GlobalMaxPooling1D()(x)
    outputs = layers.Dense(128, activation=tf.keras.activations.swish)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model