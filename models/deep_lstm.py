import tensorflow as tf


class DeepLSTM(tf.keras.Model):
    def __init__(self, embed_dim=64, lstm_units=128, num_heads=4,
                 dense_dim=32, num_transformer_blocks=2, num_channels=5):
        super(DeepLSTM, self).__init__()
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.lstm_layer = tf.layers.LSTM(lstm_units, return_sequences=True)
        self.attention = tf.layers.MultiHeadAttention(
             num_heads=num_heads, key_dim=lstm_units, dropout=0.3
         )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.layers.Dense(dense_dim, activation=tf.nn.gelu),
                tf.layers.Dense(lstm_units),
            ]
         )
        self.output_layer = tf.layers.Dense(num_channels, activation='linear')

    def call(self, inputs):
        x = tf.reshape(inputs, (tf.shape(inputs)[0], -1, self.embed_dim))
        x = self.lstm_layer(x)
        x = self.attention(x, x)
        x = tf.reshape(x, (tf.shape(x)[0], self.lstm_units))
        x = self.dense_proj(x)
        output = self.output_layer(x)
        return output


def build_model(embed_dim=64, lstm_units=128, num_heads=4,
                            dense_dim=32, num_channels=5):
    inputs = tf.keras.Input(shape=(None, embed_dim))
    x = tf.reshape(inputs, (tf.shape(inputs)[0], -1, embed_dim))
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units, dropout=0.3)(x, x)
    x = tf.keras.layers.Dense(dense_dim, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dense(lstm_units)(x)
    x = tf.reshape(x, (tf.shape(x)[0], lstm_units))
    x = tf.keras.layers.Flatten()(x)  # Flatten the output
    outputs = tf.keras.layers.Dense(num_channels, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
