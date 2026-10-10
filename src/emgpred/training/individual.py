"""Protocolo within-subject: um modelo por paciente, treino e teste no mesmo indivíduo.

Treino/validação: `train_test_split(split_test_size, random_state=42)` sobre o `_train.npz`
(padrão 80/20). Teste: `_test.npz` inteiro (trecho final de cada gravação, na ordem temporal).
Entrada: todos os 128 eletrodos ou uma seleção fixa (`data.selection` / `data.channels_file`).
"""
import os

import keras
import numpy as np
import pandas as pd


from emgpred.data.datasets import individual_split, load_split, make_dataset
from emgpred.models.augmentation import input_channels
from emgpred.training.callbacks import standard_callbacks
from emgpred.training.common import (
    build_model, compile_model, evaluate_and_save, history_info, resolve_channels,
)


def train_patient(pid, cfg):
    paths, data_cfg, train_cfg = cfg["paths"], cfg["data"], cfg["training"]
    out_dir = os.path.join(paths["save_dir"], f"paciente_{pid}")
    os.makedirs(out_dir, exist_ok=True)

    channels = resolve_channels(cfg)
    augs = cfg.get("augmentation", {}).get("names", [])
    read_ch = input_channels(augs, channels)        # com "eletrodo", lê os 128 e seleciona no modelo
    cache = paths.get("cache_dir")
    X_full, Y_full = load_split(paths["raw_dir"], pid, "train", channels=read_ch, cache_dir=cache)
    X_test, Y_test = load_split(paths["raw_dir"], pid, "test", channels=read_ch, cache_dir=cache)
    n_ch = X_test.shape[2] if channels is None else len(channels)    # eletrodos que chegam ao modelo
    tr, va = individual_split(len(X_full), test_size=data_cfg.get("split_test_size", 0.2),
                              val_fraction=data_cfg.get("val_fraction"), seed=data_cfg.get("seed", 42))
    print(f"   {n_ch} eletrodos | augmentation: {'+'.join(augs) or 'nenhuma'} | treino {len(tr):,} "
          f"| validação {len(va):,} | teste {len(X_test):,}")

    bs = train_cfg.get("batch_size", 128)
    train_ds = make_dataset(X_full, Y_full, batch_size=bs, shuffle=True, indices=tr)
    val_ds = make_dataset(X_full, Y_full, batch_size=bs, indices=np.sort(va))
    test_ds = make_dataset(X_test, batch_size=bs)

    keras.utils.set_random_seed(data_cfg.get("seed", 42))
    model = compile_model(build_model(cfg["model"], X_test.shape[1], n_ch, augmentations=augs, selection=channels,
                                      seed=data_cfg.get("seed", 42)), train_cfg)
    callbacks = standard_callbacks(out_dir, early_stop_patience=train_cfg.get("early_stop_patience", 15),
                                   lr_patience=train_cfg.get("lr_patience", 10),
                                   lr_factor=train_cfg.get("lr_factor", 0.75))
    hist = model.fit(train_ds, epochs=train_cfg.get("epochs", 400), validation_data=val_ds, verbose=1,
                     callbacks=callbacks)

    result = evaluate_and_save(model, test_ds, Y_test, pid, out_dir, kalman_kwargs=cfg.get("kalman"),
                               ma_kwargs=cfg.get("moving_average"))
    result.update(history_info(hist, callbacks[0]), train_samples=len(tr), model=cfg["model"].get("name", "gvt"),
                  params=model.count_params(), augs="+".join(augs) or "nenhuma",
                  electrodes="todos" if channels is None else " ".join(map(str, channels)))
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    return result
