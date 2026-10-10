"""Protocolo leave-one-subject-out (LOSO), corresponde ao notebook `PhD_LOSO`.

Para cada paciente retido:
- treino: `_train.npz` dos outros pacientes, menos os de validação (com `pool_test`, também o `_test.npz`);
- validação: `_test.npz` dos `val_patients` pacientes seguintes ao retido (ordem alfabética circular),
  fora do treino, para o early stopping escolher a época pela generalização a pacientes novos;
  com `val_patients = 0`, sorteio de `val_fraction` da pool;
- teste: `_test.npz` do retido (o mesmo dos treinos individuais);
- "vistos": com `eval_seen` e sem `pool_test`, o `_test.npz` de cada paciente de treino, comparável
  aos individuais (`vistos.csv` e colunas `seen_*`).
`normalize` padroniza cada canal com média/desvio do `_train` do próprio paciente (só o sinal).
`init_from` inicializa com o modelo individual de um paciente do treino deste fold.
"""
import gc
import os

import keras
import numpy as np
import pandas as pd

from emgpred.data.datasets import list_patients, load_loso_fold, load_patient_emg, make_dataset
from emgpred.models.augmentation import input_channels
from emgpred.training.callbacks import standard_callbacks
from emgpred.training.common import (
    build_model, compile_model, evaluate_and_save, history_info, regression_metrics, resolve_channels,
)


def init_from_individual(model, model_cfg, init_dir, pid, vps, patients, time_steps, n_ch, selection):
    """Copia os pesos treináveis do modelo individual (sem augmentation) do 1º paciente de treino que o tiver."""
    donors = [p for p in patients if p != pid and p not in vps
              and os.path.exists(os.path.join(init_dir, f"paciente_{p}", "model.weights.h5"))]
    if not donors:
        raise FileNotFoundError(f"nenhum modelo individual em {init_dir} para os pacientes de treino deste fold")
    ref = build_model(model_cfg, time_steps, n_ch, selection=selection)
    ref.load_weights(os.path.join(init_dir, f"paciente_{donors[0]}", "model.weights.h5"))
    weights = [w.numpy() for w in ref.trainable_weights]
    assert len(weights) == len(model.trainable_weights), "arquiteturas diferentes"
    for v, w in zip(model.trainable_weights, weights):
        assert tuple(v.shape) == w.shape, (v.path, v.shape, w.shape)
        v.assign(w)
    print(f"   🔁 inicializado com o modelo individual de {donors[0]} ({init_dir})")
    return donors[0]


def evaluate_seen(model, cfg, pid, vps, read_ch, normalize, out_dir, bs):
    """_test.npz de cada paciente de treino do fold (só válido se a pool não incluiu esses _test)."""
    paths, rows = cfg["paths"], []
    for p in list_patients(paths["raw_dir"]):
        if p == pid or p in vps:
            continue
        X, Y = load_patient_emg(paths["raw_dir"], p, "test", read_ch, normalize, paths.get("cache_dir"))
        metrics, _ = regression_metrics(Y, model.predict(make_dataset(X, batch_size=bs), verbose=0),
                                        cfg.get("kalman"), cfg.get("moving_average"))
        rows.append({"patient_id": p, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "vistos.csv"), index=False)
    print(f"   👥 vistos ({len(df)}): R² smooth {df.r2_smooth.mean():.4f} | R² média móvel {df.r2_ma.mean():.4f}")
    return {"seen_n": len(df), **{f"seen_{k}": df[k].mean() for k in df.columns if k != "patient_id"}}


def train_patient(pid, cfg):
    paths, data_cfg, train_cfg = cfg["paths"], cfg["data"], cfg["training"]
    out_dir = os.path.join(paths["save_dir"], f"paciente_{pid}")
    os.makedirs(out_dir, exist_ok=True)
    cache, seed = paths.get("cache_dir"), data_cfg.get("seed", 42)
    normalize, pool_test = data_cfg.get("normalize", False), data_cfg.get("pool_test", False)

    channels = resolve_channels(cfg)
    augs = cfg.get("augmentation", {}).get("names", [])
    read_ch = input_channels(augs, channels)        # com "eletrodo", lê os 128 e seleciona no modelo
    (X, Y, tr), (X_va, Y_va, va), vps = load_loso_fold(
        paths["raw_dir"], pid, read_ch, val_patients=data_cfg.get("val_patients", 4),
        val_fraction=data_cfg.get("val_fraction", 0.05), pool_test=pool_test, normalize=normalize,
        seed=seed, cache_dir=cache)
    X_te, Y_te = load_patient_emg(paths["raw_dir"], pid, "test", read_ch, normalize, cache)
    n_ch = X_te.shape[2] if channels is None else len(channels)
    print(f"   {n_ch} eletrodos | augmentation: {'+'.join(augs) or 'nenhuma'} | treino {len(tr):,} "
          f"| validação {len(va):,} ({'_test de ' + ', '.join(vps) if vps else 'sorteio'}) | teste {len(X_te):,} "
          f"| RAM fold {X.nbytes / 1e9:.1f} GB")

    bs = train_cfg.get("batch_size", 512)
    steps = train_cfg.get("steps_per_epoch") or len(tr) // bs   # "época" curta -> valida com mais frequência
    keras.utils.set_random_seed(seed)
    model = build_model(cfg["model"], X_te.shape[1], n_ch, augmentations=augs, selection=channels, seed=seed)
    donor = None
    if paths.get("init_from"):
        donor = init_from_individual(model, cfg["model"], paths["init_from"], pid, vps,
                                     list_patients(paths["raw_dir"]), X_te.shape[1], n_ch, channels)
    compile_model(model, train_cfg)
    val_smooth = train_cfg.get("val_smooth", 1)
    callbacks = standard_callbacks(out_dir, early_stop_patience=train_cfg.get("early_stop_patience", 15),
                                   lr_patience=train_cfg.get("lr_patience", 5),
                                   lr_factor=train_cfg.get("lr_factor", 0.75), tensorboard=False,
                                   val_smooth=val_smooth)
    hist = model.fit(make_dataset(X, Y, batch_size=bs, shuffle=True, indices=tr,
                                  repeat=bool(train_cfg.get("steps_per_epoch"))),
                     validation_data=make_dataset(X_va, Y_va, batch_size=bs, indices=va),
                     steps_per_epoch=steps, epochs=train_cfg.get("epochs", 200), verbose=1, callbacks=callbacks)
    del X, Y, X_va, Y_va
    gc.collect()

    result = evaluate_and_save(model, make_dataset(X_te, batch_size=bs), Y_te, pid, out_dir,
                               kalman_kwargs=cfg.get("kalman"), ma_kwargs=cfg.get("moving_average"))
    result.update(history_info(hist, callbacks[0], monitor="val_loss_smooth" if val_smooth > 1 else "val_loss"),
                  steps_per_epoch=steps, train_samples=len(tr), val_samples=len(va),
                  val_patients=" ".join(vps) or "sorteio", normalize=normalize, pool_test=pool_test,
                  init_from=donor or "", augs="+".join(augs) or "nenhuma",
                  electrodes="todos" if channels is None else " ".join(map(str, channels)))
    if data_cfg.get("eval_seen", True) and not pool_test:
        result.update(evaluate_seen(model, cfg, pid, vps, read_ch, normalize, out_dir, bs))
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    return result
