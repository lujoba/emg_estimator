import numpy as np
import pytest

from emgpred.data.preprocessing import clean_nan_inf, frame, normalize_0_1, normalize_minus1_1


def _clean_reference(data):
    """Implementação original (laço) do PhD_CriacaoDataset."""
    data = data.copy()
    bad = ~np.isfinite(data)
    for c in range(data.shape[1]):
        col, prob = data[:, c], bad[:, c]
        for idx in np.where(prob)[0]:
            prev = next((col[i] for i in range(idx - 1, -1, -1) if not prob[i]), None)
            nxt = next((col[i] for i in range(idx + 1, len(col)) if not prob[i]), None)
            if prev is not None and nxt is not None:
                data[idx, c] = (prev + nxt) / 2
            elif prev is not None:
                data[idx, c] = prev
            elif nxt is not None:
                data[idx, c] = nxt
            else:
                data[idx, c] = 0
    return data


def test_clean_nan_inf_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    x[rng.random(x.shape) < 0.1] = np.nan
    x[rng.random(x.shape) < 0.05] = np.inf
    x[:5, 0] = np.nan          # borda inicial
    x[-5:, 1] = -np.inf        # borda final
    x[:, 2] = np.nan           # canal inteiro inválido
    np.testing.assert_allclose(clean_nan_inf(x), _clean_reference(x))


def test_frame_matches_tf_signal_frame():
    tf = pytest.importorskip("tensorflow")
    x = np.random.default_rng(1).normal(size=(1234, 3)).astype(np.float32)
    np.testing.assert_array_equal(frame(x, 100, 10), tf.signal.frame(x, 100, 10, axis=0).numpy())


def test_normalizations():
    x = np.array([[1.0, 3.0], [5.0, 2.0]])
    assert normalize_minus1_1(x).min() == -1 and normalize_minus1_1(x).max() == 1
    assert normalize_0_1(x).min() == 0 and normalize_0_1(x).max() == 1
    np.testing.assert_array_equal(normalize_0_1(np.ones(3)), np.zeros(3))


def _fake_recording(folder, name, n_emg=40960, n_ch=4, spike_at_end=0.0, seed=0):
    """CSV de EMG (ADC de 16 bits, 2048 Hz) e de cinemática (100 Hz, colunas 4,7,10,13,16)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    emg = 32768 + rng.normal(0, 200, (n_emg, n_ch))
    emg[-2000:] += spike_at_end                        # só o fim da gravação (trecho de teste)
    pd.DataFrame(emg.round()).to_csv(folder / "emg" / name, header=False, index=False)
    n_ang = n_emg * 100 // 2048
    t = np.arange(n_ang) / 100
    ang = np.zeros((n_ang, 17))
    for j, c in enumerate([4, 7, 10, 13, 16]):
        ang[:, c] = 40 + 30 * np.sin(2 * np.pi * 0.5 * t + j)
    ang[-40:, [4, 7, 10, 13, 16]] += 10 * spike_at_end / 1000
    path = folder / "ang" / name
    path.write_text("".join(f"cabecalho {i}\n" for i in range(4)) + pd.DataFrame(ang).to_csv(index=False))


def test_normalization_uses_only_training_segments(tmp_path):
    from sklearn.model_selection import train_test_split

    from emgpred.data.datasets import build_patient_dataset

    results = {}
    for tag, spike in (("base", 0.0), ("spike", 30000.0)):
        d = tmp_path / tag
        (d / "emg").mkdir(parents=True)
        (d / "ang").mkdir()
        for task in ("1", "8"):
            _fake_recording(d, f"P_{task}_05_450_0.csv", spike_at_end=spike, seed=int(task))
        results[tag] = build_patient_dataset(str(d / "emg"), str(d / "ang"), test_size=0.1)

    (base, stats_base), (spike, stats_spike) = results["base"], results["spike"]
    # Se o teste entrasse no min/max, o pico (~5800 µV contra ~100 µV) mudaria a escala por completo.
    # Sobra só um resíduo mínimo da filtragem de fase zero (filtfilt) e do resample por FFT, que
    # são aplicados à gravação inteira antes do split.
    for key in stats_base:
        assert np.isclose(stats_spike[key], stats_base[key], rtol=1e-3), key
    np.testing.assert_allclose(base["train"][0], spike["train"][0], atol=2e-3)
    np.testing.assert_allclose(base["train"][1].astype(float), spike["train"][1].astype(float), atol=2e-3)
    assert np.abs(spike["test"][0]).max() > np.abs(base["test"][0]).max()

    # split = train_test_split(test_size=0.1, shuffle=False) por arquivo
    n_file = len(base["train"][0]) // 2 + len(base["test"][0]) // 2
    tr, te = train_test_split(np.arange(n_file), test_size=0.1, shuffle=False)
    assert len(base["train"][0]) == 2 * len(tr) and len(base["test"][0]) == 2 * len(te)
    assert base["train"][0].min() >= -1 and base["train"][0].max() <= 1
    assert np.isclose(base["train"][1].min(), 0, atol=1e-2) and np.isclose(base["train"][1].max(), 1, atol=1e-2)
