import tensorflow as tf


class AmplitudeScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scale_range=(0.8, 1.2), **kwargs):
        super(AmplitudeScalingLayer, self).__init__(**kwargs)
        self.scale_range = scale_range

    def call(self, inputs, training=False):
        if training:
            scale_factor = tf.random.uniform(shape=(1,), minval=self.scale_range[0], maxval=self.scale_range[1])
            scaled_inputs = inputs * scale_factor
            return scaled_inputs
        return inputs


class TimeShiftLayer(tf.keras.layers.Layer):
    def __init__(self, max_shift, **kwargs):
        super(TimeShiftLayer, self).__init__(**kwargs)
        self.max_shift = max_shift

    def call(self, inputs, training=False):
        if training:
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
    def __init__(self, rotation_range=(-22/7, 22/7), **kwargs):
        super(PhaseRotationLayer, self).__init__(**kwargs)
        self.rotation_range = rotation_range

    def call(self, inputs, training=False):
        if training:
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
