"""Camadas de data augmentation para janelas de EMG (ativas só com training=True).

Mesmas camadas de `emg_estimator/models/data_augmentation.py`, usadas nos notebooks da tese.

Entrada: (batch, tempo, canais) ou (batch, tempo, canais, 1); o eixo do tempo é o 1.
Os sinais têm um offset por janela/canal maior que a variação dentro da janela, então as
transformações agem sobre o desvio em torno da média temporal de cada canal e preservam esse offset.
"""
import keras
import tensorflow as tf


def _time_mean(x):
    return tf.reduce_mean(x, axis=1, keepdims=True)


def _per_sample_shape(x, per_channel=False):
    # (B, 1, 1[, 1]) ou (B, 1, C[, 1]): um valor por amostra (ou por amostra e canal)
    shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
    if per_channel:
        shape[2] = tf.shape(x)[2]
    return shape


@keras.saving.register_keras_serializable(package="emgpred")
class AmplitudeScalingLayer(keras.layers.Layer):
    """Multiplica o desvio em torno da média por um fator ~ U(scale_range), por amostra (e por canal)."""

    def __init__(self, scale_range=(0.8, 1.2), per_channel=False, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.scale_range = tuple(scale_range)
        self.per_channel = per_channel
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        scale = keras.random.uniform(_per_sample_shape(inputs, self.per_channel), self.scale_range[0],
                                     self.scale_range[1], dtype=inputs.dtype, seed=self.seed_generator)
        mean = _time_mean(inputs)
        return mean + scale * (inputs - mean)

    def get_config(self):
        return {**super().get_config(), "scale_range": self.scale_range, "per_channel": self.per_channel,
                "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class TimeShiftLayer(keras.layers.Layer):
    """Desloca cada janela por um número inteiro de amostras em [-max_shift·T, +max_shift·T].

    Sem volta circular: as amostras que entram pela borda repetem o valor da borda.
    Atenção: o alvo (ângulo) continua o mesmo, então o deslocamento deve ser pequeno.
    """

    def __init__(self, max_shift=0.05, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = max_shift
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        b, t = tf.shape(inputs)[0], tf.shape(inputs)[1]
        s = tf.cast(tf.round(self.max_shift * tf.cast(t, tf.float32)), tf.int32)
        shift = keras.random.randint((b, 1), -s, s + 1, seed=self.seed_generator)
        idx = tf.clip_by_value(tf.range(t)[None, :] - shift, 0, t - 1)          # (B, T)
        return tf.gather(inputs, idx, axis=1, batch_dims=1)

    def get_config(self):
        return {**super().get_config(), "max_shift": self.max_shift, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class PhaseRotationLayer(keras.layers.Layer):
    """Soma uma fase aleatória ~ U(rotation_range) a cada frequência, ao longo do tempo.

    A mesma rotação é usada em todos os canais (preserva a relação espacial); DC e Nyquist não mudam,
    e o espectro de magnitude é preservado. rfft/irfft garantem um sinal real.
    """

    def __init__(self, rotation_range=(-0.5, 0.5), seed=42, **kwargs):
        super().__init__(**kwargs)
        self.rotation_range = tuple(rotation_range)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x, mean = _to_time_last(inputs)
        n = tf.shape(x)[-1]
        spec = tf.signal.rfft(x)                                               # (B, C[,1], F)
        n_freq = tf.shape(spec)[-1]
        rot = keras.random.uniform(tf.stack([tf.shape(x)[0], n_freq]), self.rotation_range[0],
                                   self.rotation_range[1], seed=self.seed_generator)
        keep = tf.logical_or(tf.range(n_freq) == 0, tf.logical_and(n % 2 == 0, tf.range(n_freq) == n_freq - 1))
        rot = tf.where(keep[None, :], 0.0, rot)
        rot = tf.reshape(rot, tf.concat([[tf.shape(x)[0]], tf.ones([len(x.shape) - 2], tf.int32), [n_freq]], 0))
        spec = spec * tf.exp(tf.complex(tf.zeros_like(rot), rot))
        return _from_time_last(tf.signal.irfft(spec, fft_length=[n]), mean, inputs)

    def get_config(self):
        return {**super().get_config(), "rotation_range": self.rotation_range, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class TimeMasking3D(keras.layers.Layer):
    """Substitui um trecho contínuo de mask_factor·T amostras (posição aleatória por janela) pela média do canal."""

    def __init__(self, mask_factor=0.2, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.mask_factor = mask_factor
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        b, t = tf.shape(inputs)[0], tf.shape(inputs)[1]
        length = tf.maximum(tf.cast(tf.cast(t, tf.float32) * self.mask_factor, tf.int32), 1)
        start = keras.random.randint((b, 1), 0, t - length + 1, seed=self.seed_generator)
        pos = tf.range(t)[None, :]
        mask = tf.cast(tf.logical_and(pos >= start, pos < start + length), inputs.dtype)   # (B, T)
        mask = tf.reshape(mask, tf.concat([[b, t], tf.ones([len(inputs.shape) - 2], tf.int32)], 0))
        return inputs * (1.0 - mask) + _time_mean(inputs) * mask

    def get_config(self):
        return {**super().get_config(), "mask_factor": self.mask_factor, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class BandPassLayer(keras.layers.Layer):
    """Passa-banda aleatório por janela, aplicado na frequência (sem polos e de fase zero).

    Para cada janela sorteia f_low ~ U(low_range) e f_high ~ U(high_range) e multiplica o espectro pelo
    ganho de magnitude de um Butterworth de ordem `order`:
        |H(f)| = 1/sqrt(1 + (f_low/f)^(2n)) · 1/sqrt(1 + (f/f_high)^(2n))
    A média da janela (DC) é preservada. Resolução da FFT = fs / T (100 amostras a 2048 Hz → 20,48 Hz).
    """

    def __init__(self, low_range=(20.0, 40.0), high_range=(380.0, 450.0), fs=2048.0, order=4, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.low_range = tuple(low_range)
        self.high_range = tuple(high_range)
        self.fs = fs
        self.order = order
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def gain(self, n, f_low, f_high):
        # f_low, f_high: (B,) → ganho (B, F) nos bins da rfft
        freqs = tf.cast(tf.range(n // 2 + 1), tf.float32) * (self.fs / tf.cast(n, tf.float32))
        f = tf.maximum(freqs[None, :], 1e-6)
        hp = 1.0 / tf.sqrt(1.0 + (f_low[:, None] / f) ** (2 * self.order))
        lp = 1.0 / tf.sqrt(1.0 + (f / f_high[:, None]) ** (2 * self.order))
        g = hp * lp
        return tf.concat([tf.ones_like(g[:, :1]), g[:, 1:]], axis=1)          # DC preservado

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x, mean = _to_time_last(inputs)
        b, n = tf.shape(x)[0], tf.shape(x)[-1]
        f_low = keras.random.uniform((b,), self.low_range[0], self.low_range[1], seed=self.seed_generator)
        f_high = keras.random.uniform((b,), self.high_range[0], self.high_range[1], seed=self.seed_generator)
        g = self.gain(n, f_low, f_high)
        g = tf.reshape(g, tf.concat([[b], tf.ones([len(x.shape) - 2], tf.int32), [-1]], 0))
        spec = tf.signal.rfft(x) * tf.complex(g, tf.zeros_like(g))
        return _from_time_last(tf.signal.irfft(spec, fft_length=[n]), mean, inputs)

    def get_config(self):
        return {**super().get_config(), "low_range": self.low_range, "high_range": self.high_range,
                "fs": self.fs, "order": self.order, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class GaussianNoiseLayer(keras.layers.Layer):
    """Ruído gaussiano com SNR ~ U(snr_db_range) dB em relação ao desvio de cada canal na janela."""

    def __init__(self, snr_db_range=(20.0, 40.0), seed=42, **kwargs):
        super().__init__(**kwargs)
        self.snr_db_range = tuple(snr_db_range)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x = tf.cast(inputs, tf.float32)
        std = tf.math.reduce_std(x, axis=1, keepdims=True)                      # (B, 1, C[, 1])
        snr = keras.random.uniform(_per_sample_shape(x), self.snr_db_range[0], self.snr_db_range[1],
                                   seed=self.seed_generator)
        noise = keras.random.normal(tf.shape(x), seed=self.seed_generator) * std / 10.0 ** (snr / 20.0)
        return tf.cast(x + noise, inputs.dtype)

    def get_config(self):
        return {**super().get_config(), "snr_db_range": self.snr_db_range, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class FrequencyMaskingLayer(keras.layers.Layer):
    """Máscara de frequência (SpecAugment): zera uma faixa aleatória de largura ≤ max_width_hz dentro de band.

    Aplicada na FFT ao longo do tempo, igual em todos os canais da janela; DC preservado.
    """

    def __init__(self, max_width_hz=80.0, band=(20.0, 450.0), fs=2048.0, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.max_width_hz = max_width_hz
        self.band = tuple(band)
        self.fs = fs
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        x, mean = _to_time_last(inputs)
        b, n = tf.shape(x)[0], tf.shape(x)[-1]
        width = keras.random.uniform((b,), 0.0, self.max_width_hz, seed=self.seed_generator)
        start = keras.random.uniform((b,), self.band[0], self.band[1], seed=self.seed_generator) - width / 2
        freqs = tf.cast(tf.range(n // 2 + 1), tf.float32) * (self.fs / tf.cast(n, tf.float32))
        inside = tf.logical_and(freqs[None, :] >= start[:, None], freqs[None, :] <= (start + width)[:, None])
        g = tf.where(tf.logical_and(inside, freqs[None, :] > 0), 0.0, 1.0)
        g = tf.reshape(g, tf.concat([[b], tf.ones([len(x.shape) - 2], tf.int32), [-1]], 0))
        spec = tf.signal.rfft(x) * tf.complex(g, tf.zeros_like(g))
        return _from_time_last(tf.signal.irfft(spec, fft_length=[n]), mean, inputs)

    def get_config(self):
        return {**super().get_config(), "max_width_hz": self.max_width_hz, "band": self.band, "fs": self.fs,
                "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class TimeWarpLayer(keras.layers.Layer):
    """Estica/comprime a janela por um fator ~ U(1 - max_stretch, 1 + max_stretch), ancorado no fim da janela.

    O ângulo-alvo corresponde ao fim da janela, então a última amostra fica no lugar; o resto é reamostrado
    por interpolação linear (simula variação na velocidade de ativação). Fora da janela repete a borda.
    """

    def __init__(self, max_stretch=0.1, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.max_stretch = max_stretch
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        b, t = tf.shape(inputs)[0], tf.shape(inputs)[1]
        s = keras.random.uniform((b, 1), 1.0 - self.max_stretch, 1.0 + self.max_stretch, seed=self.seed_generator)
        last = tf.cast(t - 1, tf.float32)
        pos = last - (last - tf.cast(tf.range(t), tf.float32)[None, :]) * s       # (B, T), fim fixo
        pos = tf.clip_by_value(pos, 0.0, last)
        lo = tf.cast(tf.floor(pos), tf.int32)
        hi = tf.minimum(lo + 1, t - 1)
        w = pos - tf.cast(lo, tf.float32)
        x = tf.cast(inputs, tf.float32)
        a = tf.gather(x, lo, axis=1, batch_dims=1)
        c = tf.gather(x, hi, axis=1, batch_dims=1)
        w = tf.reshape(w, tf.concat([[b, t], tf.ones([len(inputs.shape) - 2], tf.int32)], 0))
        return tf.cast(a * (1.0 - w) + c * w, inputs.dtype)

    def get_config(self):
        return {**super().get_config(), "max_stretch": self.max_stretch, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class ChannelDropoutLayer(keras.layers.Layer):
    """Desliga cada canal com probabilidade `rate` (mau contato): o canal vira a sua média na janela."""

    def __init__(self, rate=0.1, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        drop = keras.random.uniform(_per_sample_shape(inputs, per_channel=True), seed=self.seed_generator) < self.rate
        drop = tf.cast(drop, inputs.dtype)
        return inputs * (1.0 - drop) + _time_mean(inputs) * drop

    def get_config(self):
        return {**super().get_config(), "rate": self.rate, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class ElectrodeShiftLayer(keras.layers.Layer):
    """Desloca o arranjo de eletrodos (faixa colocada fora do lugar) — entrada com os 128 canais.

    Arranjo: n_grids grades de rows × cols; dentro de cada grade, coluna k = canais k·rows .. k·rows + rows − 1
    (índices consecutivos) e colunas vizinhas a `rows` índices de distância (inferido das correlações do sinal).
    Por janela sorteia (dr, dc) em [-max_shift, max_shift]², o mesmo para todas as grades; o eletrodo (r, c)
    passa a ler o sinal de (r + dr, c + dc), repetindo a borda fora da grade.
    """

    def __init__(self, max_shift=1, rows=8, cols=8, n_grids=2, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.max_shift, self.rows, self.cols, self.n_grids = max_shift, rows, cols, n_grids
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs
        b = tf.shape(inputs)[0]
        per = self.rows * self.cols
        ch = tf.range(self.n_grids * per)
        grid, r, c = ch // per, ch % self.rows, (ch % per) // self.rows
        dr = keras.random.randint((b, 1), -self.max_shift, self.max_shift + 1, seed=self.seed_generator)
        dc = keras.random.randint((b, 1), -self.max_shift, self.max_shift + 1, seed=self.seed_generator)
        r2 = tf.clip_by_value(r[None, :] + dr, 0, self.rows - 1)
        c2 = tf.clip_by_value(c[None, :] + dc, 0, self.cols - 1)
        src = grid[None, :] * per + c2 * self.rows + r2                          # (B, 128)
        return tf.gather(inputs, src, axis=2, batch_dims=1)

    def get_config(self):
        return {**super().get_config(), "max_shift": self.max_shift, "rows": self.rows, "cols": self.cols,
                "n_grids": self.n_grids, "seed": self.seed}


@keras.saving.register_keras_serializable(package="emgpred")
class ChannelSelectLayer(keras.layers.Layer):
    """Seleciona um subconjunto fixo de canais (eixo 2) — usado depois do ElectrodeShiftLayer."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = [int(c) for c in channels]

    def call(self, inputs):
        return tf.gather(inputs, self.channels, axis=2)

    def get_config(self):
        return {**super().get_config(), "channels": self.channels}


def _to_time_last(inputs):
    # tira a média temporal e move o tempo para o último eixo (as FFTs do TF agem no último eixo)
    x = tf.cast(inputs, tf.float32)
    mean = _time_mean(x)
    perm = [0] + list(range(2, len(inputs.shape))) + [1]
    return tf.transpose(x - mean, perm), mean


def _from_time_last(x, mean, inputs):
    rank = len(inputs.shape)
    perm = [0, rank - 1] + list(range(1, rank - 1))
    return tf.cast(tf.transpose(x, perm) + mean, inputs.dtype)


# ---------------------------------------------------------------------------
# Configuração usada nos experimentos (PhD_SelecaoFixa / PhD_LOSO)
# ---------------------------------------------------------------------------
# Ordem de aplicação. "eletrodo" (ElectrodeShiftLayer) atua nos 128 canais antes da seleção de eletrodos.
AUG_NAMES = ["amplitude", "passabanda", "fase", "deslocamento", "mascara",
             "ruido", "mascara_freq", "warp", "queda_canal", "eletrodo"]


def augmentation_layer(name, seed=42):
    """Camada de augmentation pelo nome, com os parâmetros dos experimentos."""
    factories = {
        "amplitude":    lambda: AmplitudeScalingLayer(scale_range=(0.8, 1.2), per_channel=True, seed=seed),
        "passabanda":   lambda: BandPassLayer(low_range=(20.0, 40.0), high_range=(380.0, 450.0), fs=2048.0,
                                              order=4, seed=seed + 1),
        "fase":         lambda: PhaseRotationLayer(rotation_range=(-0.25, 0.25), seed=seed + 2),
        "deslocamento": lambda: TimeShiftLayer(max_shift=0.02, seed=seed + 3),          # ±2 amostras
        "mascara":      lambda: TimeMasking3D(mask_factor=0.1, seed=seed + 4),
        "ruido":        lambda: GaussianNoiseLayer(snr_db_range=(20.0, 40.0), seed=seed + 5),
        "mascara_freq": lambda: FrequencyMaskingLayer(max_width_hz=80.0, band=(20.0, 450.0), fs=2048.0,
                                                      seed=seed + 6),
        "warp":         lambda: TimeWarpLayer(max_stretch=0.1, seed=seed + 7),          # ±10% de velocidade
        "queda_canal":  lambda: ChannelDropoutLayer(rate=0.1, seed=seed + 8),           # mau contato
        "eletrodo":     lambda: ElectrodeShiftLayer(max_shift=1, seed=seed + 9),
    }
    if name not in factories:
        raise ValueError(f"augmentation desconhecida: {name} (opções: {AUG_NAMES})")
    return factories[name]()


def augmentation_tag(names):
    """Sufixo das pastas de saída, como nos notebooks: "" ou "_aug-com-amplitude-fase"."""
    names = [a for a in AUG_NAMES if a in (names or [])]
    return "_aug-com-" + "-".join(names) if names else ""


def input_channels(names, selection):
    """Canais lidos do dataset: os 128 se houver deslocamento de eletrodo (a seleção vai para dentro do modelo)."""
    return None if "eletrodo" in (names or []) else selection


def input_block(x, names, selection=None, n_channels=128, seed=42):
    """Aplica as augmentations `names` (na ordem de AUG_NAMES) à entrada simbólica `x`.

    Com "eletrodo", `x` tem os 128 canais: desloca a grade e então seleciona `selection`.
    """
    names = list(names or [])
    unknown = set(names) - set(AUG_NAMES)
    if unknown:
        raise ValueError(f"augmentations desconhecidas: {sorted(unknown)}")
    if "eletrodo" in names:
        x = augmentation_layer("eletrodo", seed)(x)
        x = ChannelSelectLayer(list(range(n_channels)) if selection is None else selection)(x)
    for name in AUG_NAMES:
        if name in names and name != "eletrodo":
            x = augmentation_layer(name, seed)(x)
    return x
