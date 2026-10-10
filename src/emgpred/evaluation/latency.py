"""Latência de inferência dos modelos (por janela e por lote), como em `cache_saliency/latencia*.py`."""
import time

import numpy as np


def benchmark(model, batch=1, reps=300, xla=True, warmup=10):
    """Mediana do tempo (ms) de `model(x, training=False)` para um lote aleatório de tamanho `batch`."""
    import tensorflow as tf

    x = tf.constant(np.random.randn(batch, *model.input_shape[1:]).astype("float32"))
    f = tf.function(lambda t: model(t, training=False), jit_compile=xla)

    def run():
        out = f(x)
        (out["angles"] if isinstance(out, dict) else out).numpy()

    for _ in range(warmup):
        run()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)
