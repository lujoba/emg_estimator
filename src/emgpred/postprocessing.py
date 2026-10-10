"""Pós-processamento das predições: suavização de Kalman e recalibração afim.

Substitui as ~25 versões de `apply_kalman` / `fit_affine` espalhadas pelos notebooks.
Todas as funções recebem arrays (T,) ou (T, n_dedos) na ordem temporal original.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Kalman (passeio aleatório 1D por dedo)
# ---------------------------------------------------------------------------
def _per_column(fn, y, **kwargs):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return fn(y, **kwargs)
    return np.stack([fn(y[:, j], **kwargs) for j in range(y.shape[1])], axis=1)


def _kalman_1d(signal, q, r, method):
    import pykalman

    kf = pykalman.KalmanFilter(
        transition_matrices=[1], observation_matrices=[1],
        initial_state_mean=signal[0], initial_state_covariance=1,
        observation_covariance=r, transition_covariance=q,
    )
    state, _ = kf.filter(signal) if method == "filter" else kf.smooth(signal)
    return state.ravel()


def kalman(y, q=0.012, r=1.0, method="smooth"):
    """Kalman com modelo de passeio aleatório.

    method='filter' é causal (usável em tempo real); 'smooth' (RTS) usa a sequência
    inteira, inclusive amostras futuras. Os padrões (q=0.012, r=1, smooth) são os
    usados nos scripts de treino; deixe explícito no texto qual variante foi usada.
    """
    if method not in ("filter", "smooth"):
        raise ValueError("method deve ser 'filter' ou 'smooth'")
    return _per_column(_kalman_1d, y, q=q, r=r, method=method)


def _fixed_lag_1d(signal, q, r, lag):
    n = len(signal)
    m_f, P_f = np.zeros(n), np.zeros(n)
    m_p, P_p = np.zeros(n), np.zeros(n)
    m_f[0], P_f[0] = signal[0], 1.0
    m_p[0], P_p[0] = m_f[0], P_f[0]
    for t in range(1, n):
        m_p[t], P_p[t] = m_f[t - 1], P_f[t - 1] + q
        K = P_p[t] / (P_p[t] + r)
        m_f[t] = m_p[t] + K * (signal[t] - m_p[t])
        P_f[t] = (1 - K) * P_p[t]

    out = np.zeros(n)
    for i in range(n):
        t_end = min(i + lag, n - 1)
        m_s, P_s = m_f[t_end], P_f[t_end]
        for k in range(t_end - 1, i - 1, -1):
            C = P_f[k] / max(P_p[k + 1], 1e-12)
            m_s = m_f[k] + C * (m_s - m_p[k + 1])
            P_s = P_f[k] + C * (P_s - P_p[k + 1]) * C
        out[i] = m_s
    return out


def fixed_lag_kalman(y, q=0.28, r=4.5, lag=10):
    """Suavizador de atraso fixo: filtro forward único + RTS limitado a `lag` amostras.

    Implementável em tempo real com atraso conhecido de `lag` amostras.
    """
    return _per_column(_fixed_lag_1d, y, q=q, r=r, lag=lag)


def windowed_kalman(y, q=0.012, r=1.0, half_window=3):
    """Kalman smoother (RTS) aplicado numa janela centrada de ±`half_window` amostras.

    Para cada instante i, roda filtro + suavização só em y[i-h : i+h+1] (estado
    inicial = primeira amostra da janela, covariância 1) e devolve a estimativa em i.
    Usa `half_window` amostras futuras, então tem atraso fixo de `half_window` janelas
    (com passo de 10 amostras a 1024 Hz, h=3 ≈ 29 ms). Bordas replicam a amostra extrema.
    """
    def _one(signal):
        h = half_window
        padded = np.pad(signal, h, mode="edge")
        W = np.lib.stride_tricks.sliding_window_view(padded, 2 * h + 1)   # (N, 2h+1)
        L = W.shape[1]
        m_f, P_f = np.zeros_like(W), np.zeros_like(W)
        m_p, P_p = np.zeros_like(W), np.zeros_like(W)
        m_prev, P_prev = W[:, 0], np.ones(len(W))
        for t in range(L):
            m_p[:, t], P_p[:, t] = (m_prev, P_prev) if t == 0 else (m_f[:, t - 1], P_f[:, t - 1] + q)
            K = P_p[:, t] / (P_p[:, t] + r)
            m_f[:, t] = m_p[:, t] + K * (W[:, t] - m_p[:, t])
            P_f[:, t] = (1 - K) * P_p[:, t]
        m_s = m_f[:, -1].copy()
        for t in range(L - 2, h - 1, -1):
            C = P_f[:, t] / P_p[:, t + 1]
            m_s = m_f[:, t] + C * (m_s - m_p[:, t + 1])
        return m_s

    return _per_column(_one, y)


def smooth(y, method="window", **kwargs):
    """Despacha para o suavizador escolhido: 'window' (Kalman ±h, padrão), 'moving_average',
    'fixed_lag', 'filter' (Kalman causal) ou 'smooth' (RTS na sequência inteira)."""
    if method == "moving_average":
        return moving_average(y, **{k: v for k, v in kwargs.items() if k in ("half_window", "causal")})
    if method == "window":
        return windowed_kalman(y, **kwargs)
    if method == "fixed_lag":
        return fixed_lag_kalman(y, **kwargs)
    return kalman(y, method=method, **kwargs)


def moving_average(y, half_window=3, causal=False):
    """Média móvel de 2h+1 amostras por soma acumulada: O(1) por amostra, bem mais barata que o Kalman.

    causal=False: janela centrada (h antes e h depois), atraso de h janelas, igual ao `windowed_kalman`.
    causal=True : só as últimas 2h+1 predições (nada do futuro), sem espera; o atraso efetivo do sinal
                  suavizado é de ~h janelas (o centro da janela), mas a saída sai no mesmo instante.
    Bordas replicam a amostra extrema.
    """
    w = 2 * half_window + 1

    def _one(signal):
        pad = (w - 1, 0) if causal else (half_window, half_window)
        c = np.concatenate([[0.0], np.cumsum(np.pad(signal, pad, mode="edge"))])
        return (c[w:] - c[:-w]) / w

    return _per_column(_one, y)


# ---------------------------------------------------------------------------
# Recalibração afim
# ---------------------------------------------------------------------------
def fit_affine(y_pred, y_ref):
    """Mínimos quadrados y_ref ≈ a * y_pred + b, por coluna -> (a, b)."""
    y_pred, y_ref = np.atleast_2d(np.asarray(y_pred).T).T, np.atleast_2d(np.asarray(y_ref).T).T
    a, b = np.zeros(y_pred.shape[1]), np.zeros(y_pred.shape[1])
    for j in range(y_pred.shape[1]):
        X = np.column_stack([y_pred[:, j], np.ones(len(y_pred))])
        a[j], b[j] = np.linalg.lstsq(X, y_ref[:, j], rcond=None)[0]
    return a, b


def apply_affine(y_pred, a, b):
    return np.asarray(y_pred) * a + b


def calibrate_affine(y_pred, y_true, calib_frac):
    """Ajusta a afim no trecho inicial (`calib_frac`) e avalia só no restante.

    Retorna (y_pred_eval_calibrado, y_true_eval). Ajustar e avaliar no mesmo trecho
    usa o rótulo do teste e superestima as métricas; por isso `calib_frac` precisa
    estar em (0, 1).
    """
    if not 0 < calib_frac < 1:
        raise ValueError("calib_frac deve estar em (0, 1): calibração e avaliação em trechos disjuntos")
    n_cal = int(len(y_true) * calib_frac)
    a, b = fit_affine(y_pred[:n_cal], y_true[:n_cal])
    return apply_affine(y_pred[n_cal:], a, b), y_true[n_cal:]
