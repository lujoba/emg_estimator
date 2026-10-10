"""Partes comuns aos protocolos de treino."""
import gc
import glob
import os

import keras
import numpy as np
import pandas as pd

from emgpred.constants import N_CHANNELS, N_FINGERS
from emgpred.data.datasets import read_channel_list
from emgpred.evaluation.metrics import compute_metrics
from emgpred.evaluation.plots import plot_predictions
from emgpred.evaluation.saliency import electrode_selection
from emgpred.models.baselines import MODELS as BASELINES
from emgpred.models.augmentation import input_block
from emgpred.models.gvt import build_gvt, gvt_backbone
from emgpred.postprocessing import moving_average, smooth


def build_model(model_cfg, time_steps, n_channels, output_dim=N_FINGERS, augmentations=None, selection=None,
                seed=42):
    """Modelo da seção [model] (`name` + hiperparâmetros), com as augmentations na entrada.

    n_channels: eletrodos que chegam ao modelo (após a seleção). Com a augmentation "eletrodo", a
    entrada tem os 128 canais e a seleção `selection` acontece dentro do modelo, depois do deslocamento.
    Sem augmentations, o modelo é exatamente o da arquitetura (mesmos pesos/nomes dos runs salvos).
    """
    cfg = dict(model_cfg)
    name = cfg.pop("name", "gvt")
    augmentations = list(augmentations or [])
    if name != "gvt" and name not in BASELINES:
        raise ValueError(f"modelo desconhecido: {name} (opções: gvt, {', '.join(BASELINES)})")
    if not augmentations:
        if name == "gvt":
            return build_gvt(time_steps=time_steps, space_dim=n_channels, channels=1, output_dim=output_dim, **cfg)
        return BASELINES[name](time_steps=time_steps, n_channels=n_channels, output_dim=output_dim, **cfg)

    n_in = N_CHANNELS if "eletrodo" in augmentations else n_channels
    inputs = keras.Input(shape=(time_steps, n_in, 1), name="emg")
    x = input_block(inputs, augmentations, selection, n_channels=N_CHANNELS, seed=seed)
    if name == "gvt":
        return keras.Model(inputs, gvt_backbone(x, time_steps, n_channels, output_dim, **cfg),
                           name="GlobalViewTransformer")
    core = BASELINES[name](time_steps=time_steps, n_channels=n_channels, output_dim=output_dim, **cfg)
    return keras.Model(inputs, core(x), name=core.name)


def make_optimizer(train_cfg):
    """Adam com clipping da norma do gradiente (sem ele o transformer às vezes diverge)."""
    return keras.optimizers.Adam(learning_rate=train_cfg.get("learning_rate", 1e-3),
                                 clipnorm=train_cfg.get("clipnorm", 1.0))


def compile_model(model, train_cfg):
    model.compile(optimizer=make_optimizer(train_cfg),
                  loss=keras.losses.Huber(delta=train_cfg.get("huber_delta", 1.0)), metrics=["mae", "mse"])
    return model


def resolve_channels(cfg):
    """Eletrodos de entrada: `data.channels_file` (lista explícita, na ordem do arquivo) ou
    `data.selection` ('todos' | 'media' | 'sharpe' | 'best'). None = todos."""
    data, paths = cfg.get("data", {}), cfg["paths"]
    if data.get("channels_file"):
        return read_channel_list(data["channels_file"])
    return electrode_selection(data.get("selection", "todos"), paths.get("saliency_dir"), data.get("k", 32),
                               best_file=paths.get("best_channels_file", "configs/channels/best32.json"),
                               random_seed=data.get("random_seed", 0))


def selection_tag(data_cfg):
    """Nome da seleção nas pastas: igual a `selection`, e `aleatorio<seed>` para o sorteio."""
    sel = data_cfg.get("selection", "todos")
    return f"aleatorio{data_cfg.get('random_seed', 0)}" if sel == "aleatorio" else sel


def save_predictions(out_dir, y_true, y_pred, name="predictions"):
    """`<name>.csv` e `ground_truth.csv`: colunas 0..4, sem suavização, na ordem do `_test.npz`."""
    pd.DataFrame(y_pred).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    pd.DataFrame(y_true).to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)


DEFAULT_MA = {"half_window": 3, "causal": False}


def regression_metrics(y_true, y_pred, kalman_kwargs=None, ma_kwargs=None):
    """R² e MAE brutos, após o Kalman configurado (`*_smooth`) e após a média móvel (`*_ma`)
    -> (dict, y_suavizado_kalman)."""
    y_smooth = smooth(y_pred, **(kalman_kwargs or {}))
    y_ma = moving_average(y_pred, **(ma_kwargs or DEFAULT_MA))
    raw, sm, ma = (compute_metrics(y_true, y) for y in (y_pred, y_smooth, y_ma))
    return {"r2_raw": raw["r2"], "r2_smooth": sm["r2"], "r2_ma": ma["r2"],
            "mae_raw": raw["mae"], "mae_smooth": sm["mae"], "mae_ma": ma["mae"]}, y_smooth


def evaluate_and_save(model, test_ds, y_test, pid, out_dir, kalman_kwargs=None, ma_kwargs=None, plot=True):
    """Prediz no teste, calcula métricas, salva CSVs e figura."""
    y_pred = model.predict(test_ds, verbose=0)
    metrics, y_smooth = regression_metrics(y_test, y_pred, kalman_kwargs, ma_kwargs)
    save_predictions(out_dir, y_test, y_pred)
    if plot:
        plot_predictions(y_test, y_smooth, f"{pid} | R² smooth={metrics['r2_smooth']:.4f}",
                         os.path.join(out_dir, "predicao.png"))
    print(f"   R² raw={metrics['r2_raw']:.4f} | R² smooth={metrics['r2_smooth']:.4f} "
          f"| MAE smooth={metrics['mae_smooth']:.4f} | R² média móvel={metrics['r2_ma']:.4f}")
    return {"patient_id": pid, **metrics, "test_samples": len(y_test)}


def history_info(history, guard=None, monitor="val_loss"):
    """Melhor época (pelo `monitor`), nº de épocas e rollbacks do CollapseGuard."""
    vals = history.history.get(monitor, [])
    return {"best_epoch": int(np.argmin(vals)) + 1 if vals else None, "epochs": len(vals),
            "rollbacks": guard.rollbacks if guard is not None else None}


def run_patients(train_fn, patients, save_dir, skip_done=True, **kwargs):
    """Executa `train_fn(pid, **kwargs)` para cada paciente e consolida `resumo_todos_pacientes.csv`."""
    results = []
    for i, pid in enumerate(patients, 1):
        if skip_done and os.path.exists(os.path.join(save_dir, f"paciente_{pid}", "results_summary.csv")):
            print(f"⏭️  {pid} já treinado")
            continue
        print(f"\n{'=' * 60}\n🏥 Paciente {pid} ({i}/{len(patients)})\n{'=' * 60}")
        results.append(train_fn(pid, **kwargs))
        keras.backend.clear_session()
        gc.collect()

    files = sorted(glob.glob(os.path.join(save_dir, "paciente_*", "results_summary.csv")))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(os.path.join(save_dir, "resumo_todos_pacientes.csv"), index=False)
    return results
