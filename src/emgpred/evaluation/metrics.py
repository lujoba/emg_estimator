"""Métricas de regressão por dedo e globais."""
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

from emgpred.constants import FINGERS


def compute_metrics(y_true, y_pred, finger_names=FINGERS):
    """R², MAE e Pearson globais e por dedo.

    O R² global é o `r2_score` multi-saída do sklearn (média uniforme dos R² por dedo);
    o Pearson global é a média dos coeficientes por dedo.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fingers = []
    for j in range(y_true.shape[1]):
        fingers.append({
            "finger": finger_names[j],
            "r2": r2_score(y_true[:, j], y_pred[:, j]),
            "mae": float(np.mean(np.abs(y_pred[:, j] - y_true[:, j]))),
            "rmse": float(np.sqrt(np.mean((y_pred[:, j] - y_true[:, j]) ** 2))),
            "pearson": stats.pearsonr(y_true[:, j], y_pred[:, j])[0],
        })
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "pearson": float(np.mean([f["pearson"] for f in fingers])),
        "fingers": fingers,
    }


def weighted_r2(y_true, y_pred):
    """R² médio ponderado pela variância de cada dedo."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r2s = [r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]
    return float(np.average(r2s, weights=np.var(y_true, axis=0)))


def flatten_metrics(metrics, prefix=""):
    """{'r2':..., 'fingers':[...]} -> dict plano para uma linha de DataFrame."""
    row = {f"{prefix}{k}": v for k, v in metrics.items() if k != "fingers"}
    for f in metrics.get("fingers", []):
        for k in ("r2", "mae", "pearson"):
            row[f"{prefix}{k}_{f['finger']}"] = f[k]
    return row
