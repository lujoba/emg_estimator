"""Autopruning de eletrodos por paciente (128 -> k), corresponde ao notebook `PhD_AutoPruning`.

Avaliação no `_test.npz` em três modos:
    dyn      seleção dinâmica por janela (precisa dos 128 gravados; limite superior)
    pat      k eletrodos fixos do paciente (implantável com k eletrodos físicos)
    pattask  k fixos por paciente e tarefa predita
As máscaras fixas vêm da frequência de seleção nas janelas de treino (só EMG). As referências
(`[references]` no config) são modelos de seleção fixa já treinados com o mesmo split.
"""
import json
import os

import keras
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax

from emgpred.constants import TASKS
from emgpred.data.datasets import individual_split, load_split, make_dataset
from emgpred.evaluation.plots import plot_electrode_selection, plot_predictions
from emgpred.evaluation.results import load_patient_predictions
from emgpred.evaluation.saliency import electrode_selection, saliency_prior
from emgpred.models.selector import build_autopruning_model, calibrate_fixed_masks, predict_angles, topk_mask
from emgpred.training.callbacks import CollapseGuard, PostPruneTraining, PruneScheduler
from emgpred.training.common import make_optimizer, regression_metrics, save_predictions

MODES = [("dyn", 0), ("pat", 1), ("pattask", 2)]
FIXED_SELECTIONS = ("sharpe", "media")


def _reference(ref_dir, pid, y_true, kalman_kwargs, ma_kwargs):
    """Métricas de um modelo de referência já treinado, no mesmo teste."""
    data = load_patient_predictions(os.path.join(ref_dir, f"paciente_{pid}"))
    if data is None:
        return None
    gt, pred = data
    if gt.shape != y_true.shape or not np.allclose(gt, y_true, atol=1e-3):
        raise ValueError(f"{ref_dir}: ground truth não confere com o _test.npz de {pid}")
    return regression_metrics(y_true, pred, kalman_kwargs, ma_kwargs)[0]


def train_patient(pid, cfg, eval_only=False):
    paths, data_cfg, train_cfg, ap = cfg["paths"], cfg["data"], cfg["training"], cfg["autopruning"]
    out_dir = os.path.join(paths["save_dir"], f"paciente_{pid}")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "model_best.weights.h5")
    k_final, seed = ap["k_final"], data_cfg.get("seed", 42)
    kalman_kwargs, ma_kwargs = cfg.get("kalman"), cfg.get("moving_average")

    prior = saliency_prior(paths["saliency_dir"], exclude_patient=pid)
    np.save(os.path.join(out_dir, "saliency_prior.npy"), prior)

    X, Y, T = load_split(paths["raw_dir"], pid, "train", with_task=True, cache_dir=paths.get("cache_dir"))
    X_te, Y_te, T_te = load_split(paths["raw_dir"], pid, "test", with_task=True, cache_dir=paths.get("cache_dir"))
    tr, va = individual_split(len(X), test_size=data_cfg.get("split_test_size", 0.2),
                              val_fraction=data_cfg.get("val_fraction"), seed=seed)
    tr, va = np.sort(tr), np.sort(va)
    print(f"   treino {len(tr):,} | validação {len(va):,} | teste {len(X_te):,} janelas")

    bs = train_cfg.get("batch_size", 128)
    test_ds = make_dataset(X_te, Y_te, T_te, batch_size=bs)
    model, gate_model, selector = build_autopruning_model(prior, time_steps=X_te.shape[1], n_ch=X_te.shape[2],
                                                          backbone=cfg["model"])
    model.compile(
        optimizer=make_optimizer(train_cfg),
        loss={"angles": keras.losses.Huber(delta=1.0),
              "task": keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
        loss_weights={"angles": 1.0, "task": ap.get("lambda_task", 0.1)},
        metrics={"angles": ["mae"], "task": ["accuracy"]},
    )

    best_epoch = rollbacks = None
    if eval_only:
        model.load_weights(weights_path)
    else:
        steps = len(tr) // bs
        assert train_cfg["epochs"] > ap["warmup_epochs"] + ap["prune_epochs"]
        scheduler = PruneScheduler(selector, k_final, ap["warmup_epochs"] * steps, ap["prune_epochs"] * steps,
                                   ap["tau_start"], ap["tau_end"])
        post = PostPruneTraining(scheduler, weights_path, lr_patience=train_cfg.get("lr_patience", 10),
                                 lr_factor=train_cfg.get("lr_factor", 0.75), min_lr=1e-8,
                                 stop_patience=train_cfg.get("early_stop_patience", 15),
                                 lr_min_delta=train_cfg.get("lr_min_delta", 1e-4))
        # set_weights também restaura k e τ do seletor: zera o cache do scheduler para ele reatribuir
        guard = CollapseGuard(monitor="val_angles_loss", on_restore=lambda: setattr(scheduler, "_last", None))
        model.fit(
            make_dataset(X, Y, T, batch_size=bs, shuffle=True, indices=tr),
            validation_data=make_dataset(X, Y, T, batch_size=bs, indices=va),
            epochs=train_cfg["epochs"], verbose=1,
            callbacks=[scheduler, guard, post,
                       keras.callbacks.CSVLogger(os.path.join(out_dir, "training_history.csv"))],
        )
        best_epoch = None if post.best_epoch is None else post.best_epoch + 1
        rollbacks = guard.rollbacks

    selector.k.assign(float(k_final))

    # --- Avaliação ---
    out_te = gate_model.predict(make_dataset(X_te, batch_size=512), verbose=0)
    sel_te = topk_mask(out_te["gate"], k_final)
    freq_task = np.stack([sel_te[T_te == t].mean(0) for t in range(len(TASKS))])
    calib = np.sort(np.random.default_rng(seed).permutation(tr)[: ap.get("calib_max", 20000)])
    masks = calibrate_fixed_masks(gate_model, X[calib], k_final, n_tasks=len(TASKS))
    fixed_sets = {name: electrode_selection(name, paths["saliency_dir"], k_final) for name in FIXED_SELECTIONS}
    pat_set = set(np.flatnonzero(masks[-1]))

    task_pred = out_te["task"].argmax(-1)
    # voto suavizado: média das probabilidades numa janela de 25 janelas (~0,25 s)
    task_smooth = uniform_filter1d(softmax(out_te["task"], axis=-1), 25, axis=0).argmax(-1)
    confusion = np.zeros((len(TASKS), len(TASKS)), int)
    np.add.at(confusion, (T_te, task_pred), 1)
    result = {
        "patient_id": pid, "best_epoch": best_epoch, "rollbacks": rollbacks,
        "task_acc": float(np.mean(task_pred == T_te)), "task_acc_smooth": float(np.mean(task_smooth == T_te)),
        "alpha": float(selector.alpha.numpy()), "train_samples": len(tr), "test_samples": len(X_te),
        "dyn_unique_electrodes": int((sel_te.sum(0) > 0).sum()),
    }
    for name, sel in fixed_sets.items():
        result[f"jaccard_pat_vs_{name}"] = len(pat_set & set(sel)) / len(pat_set | set(sel))

    smooth_by_mode = {}
    for name, mode in MODES:
        preds = predict_angles(model, selector, test_ds, mode, masks)
        metrics, y_smooth = regression_metrics(Y_te, preds, kalman_kwargs, ma_kwargs)
        smooth_by_mode[name] = (preds, y_smooth)
        result.update({f"{name}_{k}": v for k, v in metrics.items()})
    for ref_name, ref_dir in cfg.get("references", {}).items():
        ref = _reference(ref_dir, pid, Y_te, kalman_kwargs, ma_kwargs)
        if ref:
            result.update({f"{ref_name}_{k}": v for k, v in ref.items()})

    for name in [*cfg.get("references", {}), *(m for m, _ in MODES)]:
        if f"{name}_r2_raw" in result:
            print(f"   [{name:8s}] R² raw={result[f'{name}_r2_raw']:.4f} | "
                  f"R² smooth={result[f'{name}_r2_smooth']:.4f} | MAE smooth={result[f'{name}_mae_smooth']:.4f} "
                  f"| R² média móvel={result[f'{name}_r2_ma']:.4f}")

    # --- Salvar (colunas 0..4, sem suavização, ordem do _test.npz) ---
    for name, (preds, _) in smooth_by_mode.items():
        save_predictions(out_dir, Y_te, preds, name=f"predictions_{name}")
    cols = [f"E_{j}" for j in range(X_te.shape[2])]
    pd.DataFrame(freq_task, index=TASKS, columns=cols).to_csv(os.path.join(out_dir, "selecao_dinamica_freq_por_tarefa.csv"))
    pd.DataFrame(masks.astype(int), index=TASKS + ["paciente"], columns=cols).to_csv(
        os.path.join(out_dir, "mascaras_fixas.csv"))
    with open(os.path.join(out_dir, "eletrodos_selecionados.json"), "w") as f:
        json.dump({row: np.flatnonzero(m).tolist() for row, m in zip(TASKS + ["paciente"], masks)}, f, indent=1)
    pd.DataFrame(confusion, index=TASKS, columns=TASKS).to_csv(os.path.join(out_dir, "tarefa_confusao.csv"))
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    plot_predictions(Y_te, smooth_by_mode["pat"][1],
                     f"Autopruning ({k_final} fixos/paciente) — {pid} | R²={result['pat_r2_smooth']:.4f}",
                     os.path.join(out_dir, "predicao_pat.png"))
    plot_electrode_selection(freq_task, masks, fixed_sets, os.path.join(out_dir, "selecao_eletrodos.png"))
    return result
