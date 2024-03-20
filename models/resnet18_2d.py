import tensorflow as tf
import keras
from keras import layers


# class Conv2Plus1D(layers.Layer):
#   def __init__(self, filters, kernel_size, padding):
#     """
#       A sequence of convolutional layers that first apply the convolution
#       operation over the spatial dimensions, and then the temporal dimension. 
#     """
#     super().__init__()
#     self.seq = keras.Sequential([  
#         # Spatial decomposition
#         layers.Conv2D(filters=filters,
#                       kernel_size=(1, kernel_size[1]),
#                       padding=padding),
#         # Temporal decomposition
#         layers.Conv2D(filters=filters, 
#                       kernel_size=(kernel_size[0], 1),
#                       padding=padding)
#         ])
# 
#   def call(self, x):
#     return self.seq(x)


class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization,
    and the activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'), # Conv2Plus1D
        layers.LayerNormalization(),
        layers.ReLU(),
        layers.Conv2D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'), # Conv2Plus1D
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

    def call(self, signal):
        """
        Redimensiona um tensor 2D.

        Args:
            signal: Tensor de entrada.

        Retorna:
            Tensor redimensionado.
        """
        resized_signal = tf.image.resize(signal, (self.height, self.width))
        return resized_signal

def build_model(sequence_length=100, embed_dim=256, complex=False):
    input_shape = (None, sequence_length, embed_dim, 1)
    input = layers.Input(shape=(input_shape[1:]))
    x = input
    ######################################################
    # Feature extraction filter
    ######################################################
    x = layers.Conv2D(filters=16, kernel_size=(3, 7), padding='same')(x) # Conv2Plus1D
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeSignal(-(embed_dim // -2), sequence_length // 2)(x)
    # Block 1
    x = add_residual_block(x, 16, (3, 3))
    x = ResizeSignal(-(embed_dim // -4), sequence_length // 4)(x)
    # Block 2
    x = add_residual_block(x, 32, (3, 3))
    x = ResizeSignal(-(embed_dim // -8), sequence_length // 8)(x)
    # Block 3
    x = add_residual_block(x, 64, (3, 3))
    x = ResizeSignal(-(embed_dim // -16), sequence_length // 16)(x)
    # Block 4
    x = add_residual_block(x, embed_dim, (3, 3))

    # Features Block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    ######################################################
    # Linear Combinator
    ######################################################
    x = layers.Dense(embed_dim)(x)
    ######################################################
    # Model Building
    ######################################################
    model = keras.Model(input, x)
    
    return model
