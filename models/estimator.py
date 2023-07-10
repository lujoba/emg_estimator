import tensorflow as tf
import einops
import keras
from keras import layers

class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and
    the activation function, ReLU.
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
    # x = tf.expand_dims(x, -2)  # Add a new channel axis
    return self.seq(x)


class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through
    different sized filters and downsampled.
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
    Add residual blocks to the model. If the last dimensions of the input
    data and filter size does not match, project it such that last
    dimension matches.
  """
  out = ResidualMain(filters,
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the
  # tensor to match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeSignal(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, signal):
    """
      Use the einops library to resize the tensor.

      Args:
        Signal: Tensor representation of the signal, in the form of a set
        of frames.

      Return:
        A downsampled size of the signal according to the new height and
        width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height,
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(signal, 'b t h w c')
    images = einops.rearrange(signal, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    signals = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return signals

class Conv2Plus1D(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        padding,
        activation='linear',
        dilation_rate=1,
        **kwargs
        ):
        """
          A sequence of convolutional layers that first apply the convolution
          operation over the spatial dimensions, and then the temporal
          dimension.
        """
        super().__init__(**kwargs)
        self.spatial_conv = layers.Conv3D(
            filters=filters,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            padding=padding,
            activation=activation,
            dilation_rate=(1, dilation_rate, dilation_rate)
            )
        self.temporal_conv = layers.Conv3D(
            filters=filters,
            kernel_size=(kernel_size[0], 1, 1),
            padding=padding,
            activation=activation,
            dilation_rate=(dilation_rate, 1, 1)
            )
        self.batch_norm = layers.BatchNormalization()

    def call(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.batch_norm(x)
        x = tf.keras.activations.relu(x)
        return x

class DilatedConv2Plus1D(tf.keras.layers.Layer):
    """
    The dilation rate is set as an attribute and determines the spacing
     between the kernel elements during convolution, allowing for an
     increased receptive field.
    """
    def __init__(self, dilation_rate, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv2plus1d = Conv2Plus1D(
            filters=filters, kernel_size=kernel_size, padding="same"
            )
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dilation_rate = dilation_rate  # save dilation_rate as an attribute

    def call(self, x):
        x = self.conv2plus1d(x)
        x = self.batchnorm(x)
        x = tf.keras.activations.relu(x)
        return x


# class Estimator(tf.keras.Model):
#     """
#     The WaveNet class is a neural network model that uses the
#     DilatedConv2Plus1D class to build WaveNet blocks with a variable number
#     of layers and dilations
#     """
# 
#     def __init__(self, num_blocks, num_layers, filters, kernel_size, dilations, input_shape_, num_lstm_blocks, **kwargs):
#         super(Estimatior, self).__init__(**kwargs)
#         self.num_blocks = num_blocks
#         self.num_layers = num_layers
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.dilations = dilations
#         self.input_shape_ = input_shape_
#         self.num_lstm_blocks = num_lstm_blocks
# 
#         # Build the model
#         self.build_model()
# 
#     def build_model(self):
#         # Define input layer
#         inputs = tf.keras.Input(shape=self.input_shape_)
# 
#         # Build WaveNet blocks
#         x = inputs
#         for i in range(self.num_blocks):
#             for j in range(self.num_layers):
#                 dilation_rate = self.dilations[j % len(self.dilations)]
#                 x = DilatedConv2Plus1D(dilation_rate=dilation_rate, filters=self.filters, kernel_size=self.kernel_size)(x)
#         x = tf.keras.layers.ELU()(x)
# 
#         # Add residual blocks
#         for i in range(self.num_blocks):
#             x = add_residual_block(x, self.filters, (18, 3, 3))
#             x = ResizeSignal(x.shape[2] // 4, x.shape[3] // 4)(x)
#         x = tf.keras.layers.GlobalAveragePooling3D()(x)
#         x = tf.keras.layers.Flatten()(x)
# 
#         # Output layer
#         outputs = tf.keras.layers.Dense(1, activation="linear")(x)
# 
#         # Create the model
#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 
#     def call(self, inputs):
#         return self.model(inputs)
# 
#     def summary(self):
#         self.model.summary()
# 
#     def plot_model(self, show_shapes=False):
#         tf.keras.utils.plot_model(
#             self.model,
#             show_shapes=show_shapes
#         )


def estimator(
    num_blocks,
    num_layers,
    filters,
    kernel_size,
    dilations,
    input_shape_,
    # num_lstm_blocks,
    ):
    """
    The WaveNet function is a neural network model that uses the
    DilatedConv2Plus1D class to build WaveNet blocks with a variable number
    of layers and dilations
    """
    # Define input layer
    input_layer = layers.Input(shape=input_shape_)

    # Build WaveNet blocks
    x = input_layer
    for i in range(num_blocks):
        for j in range(num_layers):
            dilation_rate = dilations[j % len(dilations)]
            x = DilatedConv2Plus1D(
                dilation_rate=dilation_rate,
                filters=filters,
                kernel_size=kernel_size
            )(x)
    x = layers.ELU()(x)
    # Add residual blocks
    for i in range(num_blocks):
        x = add_residual_block(x, filters, (18, 3, 3))
        x = ResizeSignal(x.shape[2] // 4, x.shape[3] // 4)(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.ReLU()(x)
    # Linear combinator layer
    x = layers.Dense(1, activation="linear")(x)
    # Create the Keras model
    model = tf.keras.Model(inputs=input_layer, outputs=x)

    return model
