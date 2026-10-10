import numpy as np
import pytest

from emgpred.postprocessing import (
    apply_affine, calibrate_affine, fit_affine, fixed_lag_kalman, kalman, windowed_kalman,
)


def test_affine_recovers_gain_and_bias():
    x = np.random.default_rng(0).normal(size=(500, 5))
    a, b = fit_affine(x, 2 * x + 0.5)
    np.testing.assert_allclose(a, 2, atol=1e-10)
    np.testing.assert_allclose(b, 0.5, atol=1e-10)
    np.testing.assert_allclose(apply_affine(x, a, b), 2 * x + 0.5)


def test_calibrate_affine_uses_disjoint_segments():
    y = np.arange(100.0)[:, None]
    pred, true = calibrate_affine(y, y, 0.25)
    assert len(true) == 75 and true[0, 0] == 25
    with pytest.raises(ValueError):
        calibrate_affine(y, y, 1.0)


def test_kalman_matches_pykalman_direct():
    pykalman = pytest.importorskip("pykalman")
    s = np.cumsum(np.random.default_rng(1).normal(size=400)) * 0.01
    kf = pykalman.KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=s[0],
                               initial_state_covariance=1, observation_covariance=1, transition_covariance=0.012)
    np.testing.assert_allclose(kalman(s), kf.smooth(s)[0].ravel())
    np.testing.assert_allclose(kalman(s, method="filter"), kf.filter(s)[0].ravel())
    assert kalman(np.stack([s, s], 1)).shape == (400, 2)


def test_fixed_lag_has_bounded_lookahead():
    s = np.random.default_rng(2).normal(size=200)
    out = fixed_lag_kalman(s, q=0.1, r=1.0, lag=5)
    s2 = s.copy()
    s2[150:] += 10.0                       # muda só o futuro a partir de t=150
    out2 = fixed_lag_kalman(s2, q=0.1, r=1.0, lag=5)
    np.testing.assert_allclose(out[:145], out2[:145])   # saídas até 144 não enxergam t>=150
    assert not np.allclose(out[145:150], out2[145:150])


def test_windowed_kalman_matches_pykalman_per_window():
    pytest.importorskip("pykalman")
    s = np.cumsum(np.random.default_rng(3).normal(size=60)) * 0.05
    out = windowed_kalman(s, q=0.012, r=1.0, half_window=3)
    for i in (3, 20, 56):                  # pontos internos (sem padding)
        ref = kalman(s[i - 3:i + 4], q=0.012, r=1.0, method="smooth")[3]
        np.testing.assert_allclose(out[i], ref, atol=1e-10)


def test_non_inferiority():
    from emgpred.evaluation.stats import non_inferiority

    rng = np.random.default_rng(0)
    b = rng.random(20)
    assert non_inferiority(b - 0.001, b, margin=0.01)["verdict"] == "não inferior"
    assert non_inferiority(b - 0.05, b, margin=0.01)["verdict"] == "inconclusivo"


def test_moving_average_centered_and_causal():
    from emgpred.postprocessing import moving_average, smooth

    s = np.random.default_rng(4).normal(size=(200, 5))
    c = moving_average(s, half_window=3)
    np.testing.assert_allclose(c[10], s[7:14].mean(0))            # centrada: i-3..i+3
    k = moving_average(s, half_window=3, causal=True)
    np.testing.assert_allclose(k[10], s[4:11].mean(0))            # causal: i-6..i
    s2 = s.copy()
    s2[100:] += 5.0
    np.testing.assert_allclose(moving_average(s2, 3, causal=True)[:100], k[:100])   # não vê o futuro
    np.testing.assert_allclose(smooth(s, method="moving_average", half_window=3, q=1.0), c)
    assert c.shape == s.shape and k.shape == s.shape
