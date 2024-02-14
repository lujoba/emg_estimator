import tensorflow as tf


class AmplitudeScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scale_range=(0.8, 1.2), seed=42, **kwargs):
        super(AmplitudeScalingLayer, self).__init__(**kwargs)
        self.scale_range = scale_range
        self.seed = seed
        
    def normalize(self, tensor):
        return tf.divide(
            tf.subtract(
                tensor,
                tf.reduce_min(tensor)
            ),
            tf.subtract(
                tf.reduce_max(tensor),
                tf.reduce_min(tensor)
            )
        )

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        if training:
            tf.random.set_seed(self.seed)
            scale_factor = tf.random.uniform(
                shape=(1,),
                minval=self.scale_range[0],
                maxval=self.scale_range[1]
            )
            scaled_inputs = inputs * scale_factor
            return scaled_inputs
        return inputs


class TimeShiftLayer(tf.keras.layers.Layer):
    def __init__(self, max_shift, seed=42, **kwargs):
        super(TimeShiftLayer, self).__init__(**kwargs)
        self.max_shift = max_shift
        self.seed = seed

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        if training:
            tf.random.set_seed(self.seed)
            # Gera um fator de deslocamento aleatório dentro do intervalo permitido
            shift_factor = tf.random.uniform(shape=(), minval=-self.max_shift, maxval=self.max_shift)

            # Calcula o número de amostras para deslocar
            num_samples = tf.cast(tf.shape(inputs)[1], tf.float32)
            shift_samples = tf.cast(tf.round(shift_factor * num_samples), tf.int32)

            # Realiza o deslocamento temporal usando tf.roll
            shifted_inputs = tf.roll(inputs, shift=shift_samples, axis=1)

            return shifted_inputs
        return inputs


class PhaseRotationLayer(tf.keras.layers.Layer):
    def __init__(self, rotation_range, seed=42, **kwargs):
        super(PhaseRotationLayer, self).__init__(**kwargs)
        self.rotation_range = rotation_range
        self.seed = seed

    def get_config(self):
        config = super().get_config()
        config.update({
            "rotation_range": self.rotation_range,
        })
        return config

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        if training:
            # Seta a seed para a reprodutibilidade
            tf.random.set_seed(self.seed)

            # Aplica a transformada de Fourier no sinal de entrada
            frequency_domain = tf.signal.fft(tf.cast(inputs, tf.complex64))

            # Obtém a magnitude e a fase dos componentes de frequência
            magnitude = tf.abs(frequency_domain)
            phase = tf.math.angle(frequency_domain)

            # Gera uma rotação de fase aleatória para cada componente de frequência
            rotation_factor = tf.random.uniform(shape=tf.shape(phase),
                                                minval=self.rotation_range[0],
                                                maxval=self.rotation_range[1])

            # Aplica a rotação de fase
            rotated_phase = phase + rotation_factor

            # Reconstrói os componentes de frequência com a magnitude original e a fase rotacionada
            rotated_frequency_domain = tf.complex(magnitude * tf.cos(rotated_phase),
                                                  magnitude * tf.sin(rotated_phase))

            # Aplica a transformada inversa de Fourier para obter o sinal EMG rotacionado
            rotated_signal = tf.signal.ifft(rotated_frequency_domain)

            return tf.cast(tf.math.real(rotated_signal), inputs.dtype)
        return inputs


class TimeMasking3D(tf.keras.layers.Layer):
    def __init__(self, mask_factor=0.2, seed=42):
        super(TimeMasking3D, self).__init__()
        self.mask_factor = mask_factor
        self.seed = seed

    # @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        if training:
            # Seta a seed para a reprodutibilidade
            tf.random.set_seed(self.seed)
    
            batch_size = tf.shape(inputs)[0]
            num_time_steps = tf.shape(inputs)[1]
            mask_length = tf.cast(tf.cast(num_time_steps, tf.float32) * self.mask_factor, tf.int32)
    
            # Generate random mask positions
            mask_positions = tf.random.uniform(shape=(batch_size,), minval=0, maxval=num_time_steps-mask_length, dtype=tf.int32)
    
            # Create a time mask tensor
            mask = tf.one_hot(mask_positions, num_time_steps)
            mask = tf.cast(mask, inputs.dtype)
            mask = tf.reshape(mask, (batch_size, num_time_steps, 1, 1, 1))
    
            # Apply time masking by element-wise multiplication
            return inputs * (1.0 - mask)
        return inputs
