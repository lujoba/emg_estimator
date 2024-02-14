import einops
import tensorflow as tf
import keras
from keras import layers


class Conv2Plus1D(layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution
      operation over the spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv2D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv2D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)


class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization,
    and the activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)


class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)


def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeSignal(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        Signal: Tensor representation of the signal, in the form of a set of frames.

      Return:
        A downsampled size of the signal according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, head_size=256, num_heads=4, ff_dim=4, dropout=0):
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def call(self, inputs):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

# def transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0):
#     # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs
# 
#     # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

def build_model(WINDOW_SIZE, SENSORS_NUMBER, num_lstm_blocks, output_dim):
    input_shape = (None, WINDOW_SIZE, SENSORS_NUMBER, 1)
    input = layers.Input(shape=(input_shape[1:]))
    x = input
    ######################################################
    # Feature extraction filter
    ######################################################
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeSignal(-(HEIGHT // -2), WIDTH // 2)(x)
    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeSignal(-(HEIGHT // -4), WIDTH // 4)(x)
    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeSignal(-(HEIGHT // -8), WIDTH // 8)(x)
    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeSignal(-(HEIGHT // -16), WIDTH // 16)(x)
    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    # Features Block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    # LSTM Block
    for _ in range(num_lstm_blocks):
        x = tf.expand_dims(x, -1)
        x = layers.LSTM(128)(x)
    featuers = x
    ######################################################
    # Linear Combinator
    ######################################################
    features = layers.Dense(128)(x)
    x = layers.Dense(output_dim)(features)
    ######################################################
    # Model Building
    ######################################################
    model = keras.Model(input, x)
    features_extractor = keras.Model(input, features)
    features_input = keras.Input(shape=(128,))
    linear_combinator_layer = model.layers[-1]
    linear_combinator = keras.Model(
    features_input, linear_combinator_layer(features_input)
    )
    return model, features_extractor, linear_combinator
