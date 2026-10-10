"""Busca da combinação de data augmentation (notebook `PhD_AugSearch`).

Treina o modelo individual em `tune_patients` com diferentes combinações de `AUG_NAMES` e escolhe a de
menor perda de validação. A validação é temporal (os 20% finais de cada gravação do `_train.npz`, com
`gap_windows` de folga), então o `_test.npz` não participa da escolha.

- greedy: começa sem augmentation e, a cada rodada, acrescenta a camada que mais reduz a perda; para
  quando nenhuma reduz pelo menos `min_rel_gain` (relativo).
- exhaustive: testa todas as combinações.

Cada tentativa vai para `<search_dir>/trials.csv` (tentativas já feitas são reaproveitadas) e a escolha
para `escolhida.json`. O treino final dos 21 pacientes é o `emgpred train individual` com
`augmentation.names` = a combinação escolhida.
"""
import gc
import json
import os
import time
from itertools import combinations

import keras
import numpy as np
import pandas as pd

from emgpred.data.datasets import load_split, make_dataset, recording_ids, temporal_split
from emgpred.models.augmentation import AUG_NAMES, input_channels
from emgpred.training.callbacks import standard_callbacks
from emgpred.training.common import build_model, compile_model, resolve_channels


def _key(augs):
    return "+".join(a for a in AUG_NAMES if a in augs) or "nenhuma"


class AugSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        s = cfg["search"]
        self.tune_patients = s.get("tune_patients", ["BBS"])
        self.candidates = s.get("candidates", AUG_NAMES)
        self.search_dir = cfg["paths"]["search_dir"]
        self.trials_csv = os.path.join(self.search_dir, "trials.csv")
        os.makedirs(self.search_dir, exist_ok=True)

    def trials(self):
        if os.path.exists(self.trials_csv):
            return pd.read_csv(self.trials_csv)
        return pd.DataFrame(columns=["patient_id", "augs", "val_loss", "best_epoch", "epochs", "rollbacks", "minutes"])

    def run_trial(self, pid, augs):
        key, done = _key(augs), self.trials()
        hit = done[(done.patient_id == pid) & (done.augs == key)]
        if len(hit):
            return float(hit.val_loss.iloc[0])
        cfg, paths, s = self.cfg, self.cfg["paths"], self.cfg["search"]
        print(f"\n🔎 {pid} | augs = {key}")
        out_dir = os.path.join(self.search_dir, pid, key)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()

        channels = resolve_channels(cfg)
        read_ch = input_channels(augs, channels)
        X, Y = load_split(paths["raw_dir"], pid, "train", channels=read_ch, cache_dir=paths.get("cache_dir"))
        tr, va = temporal_split(recording_ids(paths["raw_dir"], pid), s.get("val_fraction", 0.2),
                                s.get("gap_windows", 4))
        n_ch = X.shape[2] if channels is None else len(channels)
        seed, bs = cfg["data"].get("seed", 42), cfg["training"].get("batch_size", 128)
        keras.utils.set_random_seed(seed)
        model = compile_model(build_model(cfg["model"], X.shape[1], n_ch, augmentations=augs, selection=channels,
                                          seed=seed), cfg["training"])
        callbacks = standard_callbacks(out_dir, early_stop_patience=cfg["training"].get("early_stop_patience", 15),
                                       lr_patience=cfg["training"].get("lr_patience", 10),
                                       lr_factor=cfg["training"].get("lr_factor", 0.75), tensorboard=False)
        hist = model.fit(make_dataset(X, Y, batch_size=bs, shuffle=True, indices=tr),
                         validation_data=make_dataset(X, Y, batch_size=bs, indices=va),
                         epochs=cfg["training"].get("epochs", 400), verbose=2, callbacks=callbacks)
        v = hist.history["val_loss"]
        row = {"patient_id": pid, "augs": key, "val_loss": float(np.min(v)), "best_epoch": int(np.argmin(v)) + 1,
               "epochs": len(v), "rollbacks": callbacks[0].rollbacks, "minutes": (time.time() - t0) / 60}
        pd.concat([self.trials(), pd.DataFrame([row])], ignore_index=True).to_csv(self.trials_csv, index=False)
        print(f"   val_loss = {row['val_loss']:.6f} (época {row['best_epoch']}) | {row['minutes']:.1f} min")
        del model, X, Y
        keras.backend.clear_session()
        gc.collect()
        return row["val_loss"]

    def score(self, augs):
        """Perda de validação relativa à sem augmentation, média entre os pacientes de ajuste."""
        return float(np.mean([self.run_trial(p, augs) / self.run_trial(p, []) for p in self.tune_patients]))

    def search(self):
        s = self.cfg["search"]
        if s.get("method", "greedy") == "exhaustive":
            combos = [list(c) for k in range(len(self.candidates) + 1) for c in combinations(self.candidates, k)]
            scores = {_key(c): self.score(c) for c in combos}
            best = min(scores, key=scores.get)
            chosen = [] if best == "nenhuma" else best.split("+")
        else:
            chosen, best_score = [], self.score([])
            min_gain = s.get("min_rel_gain", 0.01)
            while len(chosen) < len(self.candidates):
                cand = {a: self.score(chosen + [a]) for a in self.candidates if a not in chosen}
                a = min(cand, key=cand.get)
                print(f"\n➡️  rodada {len(chosen) + 1}: " + " | ".join(f"+{k} {v:.4f}" for k, v in cand.items())
                      + f" | atual {best_score:.4f}")
                if cand[a] > best_score * (1 - min_gain):
                    break
                chosen.append(a)
                best_score = cand[a]
        chosen = [a for a in AUG_NAMES if a in chosen]
        print(f"\n🏁 combinação escolhida: {chosen or 'nenhuma'}")
        print(self.trials().sort_values("val_loss").to_string(index=False))
        with open(os.path.join(self.search_dir, "escolhida.json"), "w") as f:
            json.dump({"augs": chosen, "search": s.get("method", "greedy"), "tune_patients": self.tune_patients}, f)
        return chosen
