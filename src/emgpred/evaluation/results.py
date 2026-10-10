"""Leitura dos resultados salvos por paciente (`<run_dir>/paciente_<PID>/`) e consolidação.

Convive com as duas convenções de nome usadas nos treinos:
    predictions.csv / ground_truth.csv                 (treino individual)
    predictions_<PID>.csv / ground_truth_<PID>.csv     (LOSO, autopruning)
"""
import os

import numpy as np
import pandas as pd

from emgpred.evaluation.metrics import compute_metrics, flatten_metrics
from emgpred.postprocessing import calibrate_affine, moving_average, smooth


def patient_dirs(run_dir):
    return sorted(d for d in os.listdir(run_dir)
                  if d.startswith("paciente_") and os.path.isdir(os.path.join(run_dir, d)))


def _first_existing(folder, names):
    for name in names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def read_predictions_csv(path):
    """Lê um CSV de predições/ground truth salvo com `DataFrame.to_csv(index=False)`.

    O cabeçalho (`0,1,2,3,4` ou `Predicted_Finger_i`) é descartado. Ler com
    `header=None` transforma esse cabeçalho numa amostra a mais, idêntica em
    predição e ground truth, o que altera o R².
    """
    return pd.read_csv(path, header=0).to_numpy(np.float64)


def load_patient_predictions(folder, pred_name=None, gt_name=None):
    """-> (ground_truth, predictions) do mesmo comprimento, ou None se faltar arquivo."""
    pid = os.path.basename(folder).replace("paciente_", "")
    with_pid = lambda name: name.replace(".csv", f"_{pid}.csv")     # execuções antigas: <nome>_<PID>.csv
    pred_name, gt_name = pred_name or "predictions.csv", gt_name or "ground_truth.csv"
    pred = _first_existing(folder, [pred_name, with_pid(pred_name)])
    gt = _first_existing(folder, [gt_name, with_pid(gt_name)])
    if pred is None or gt is None:
        return None
    y_true, y_pred = read_predictions_csv(gt), read_predictions_csv(pred)
    n = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        print(f"  ⚠️ {pid}: ground truth ({len(y_true)}) e predições ({len(y_pred)}) com tamanhos diferentes")
    return y_true[:n], y_pred[:n]


def evaluate_patient(y_true, y_pred, kalman_kwargs=None, affine_calib_frac=None, ma_kwargs=None):
    """Métricas por estágio: bruto, +Kalman, +média móvel e +Kalman+afim (opcional).

    A afim é ajustada no trecho inicial `affine_calib_frac` e todas as métricas do
    estágio 'affine' são calculadas só no trecho restante.
    """
    y_smooth = smooth(y_pred, **(kalman_kwargs or {}))
    out = {"raw": compute_metrics(y_true, y_pred), "kalman": compute_metrics(y_true, y_smooth),
           "ma": compute_metrics(y_true, moving_average(y_pred, **(ma_kwargs or {"half_window": 3})))}
    if affine_calib_frac is not None:
        y_cal, y_true_eval = calibrate_affine(y_smooth, y_true, affine_calib_frac)
        out["affine"] = compute_metrics(y_true_eval, y_cal)
    return out


def collect_results(run_dir, kalman_kwargs=None, affine_calib_frac=None, ma_kwargs=None, **load_kwargs):
    """Uma linha por paciente com as métricas de cada estágio (prefixos raw_/kalman_/ma_/affine_)."""
    rows = []
    for d in patient_dirs(run_dir):
        data = load_patient_predictions(os.path.join(run_dir, d), **load_kwargs)
        if data is None:
            print(f"  AVISO: predições não encontradas em {d}")
            continue
        stages = evaluate_patient(*data, kalman_kwargs=kalman_kwargs, affine_calib_frac=affine_calib_frac,
                                  ma_kwargs=ma_kwargs)
        row = {"patient": d.replace("paciente_", "")}
        for stage, metrics in stages.items():
            row.update(flatten_metrics(metrics, prefix=f"{stage}_"))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["patient"])
    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)


def summarize(df, columns=None):
    """média ± desvio das colunas numéricas (ou das `columns` indicadas)."""
    columns = columns or [c for c in df.columns if c != "patient" and pd.api.types.is_numeric_dtype(df[c])]
    return pd.DataFrame({"mean": df[columns].mean(), "std": df[columns].std()})
