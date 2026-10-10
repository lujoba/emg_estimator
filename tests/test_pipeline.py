"""Smoke test ponta a ponta com dados sintéticos (2 épocas, modelo pequeno)."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from emgpred.constants import TASKS  # noqa: E402
from emgpred.evaluation.results import collect_results  # noqa: E402


def _fake_raw(tmp_path, pids=("AAA", "BBB"), n=96, t=20, c=12):
    rng = np.random.default_rng(0)
    raw = tmp_path / "RAW"
    raw.mkdir()
    for pid in pids:
        for split, m in (("train", n), ("test", n // 2)):
            task = [str(1 + i % 8) for i in range(m)]
            np.savez_compressed(raw / f"{pid}_dataset_{split}.npz",
                                emg=rng.normal(size=(m, t, c)).astype(np.float16),
                                angles=rng.random((m, 1, 5)).astype(np.float16),
                                metadata={"task_id": task})
    return raw


def _fake_saliency(tmp_path, n_ch=12, k=4):
    sal = tmp_path / "sal"
    sal.mkdir()
    cols = [f"E_{j}" for j in range(n_ch)]
    for name in TASKS:
        pd.DataFrame(np.random.default_rng(1).random((2, n_ch)), index=["paciente_AAA", "paciente_BBB"],
                     columns=cols).to_csv(sal / f"tabela_importancia_movimento_{name}.csv")
    pd.DataFrame({"ImportanciaMediaGlobal": np.arange(n_ch, dtype=float)}, index=cols).to_csv(
        sal / "tabela_importancia_global.csv")
    pd.DataFrame({"EletrodosSelecionados": [f"E_{j}" for j in range(k)]}).to_csv(
        sal / f"melhores_{k}_eletrodos.csv", index=False)
    return sal


def _cfg(tmp_path, raw):
    small = dict(d_model=16, n_heads=2, n_layers_time=1, n_layers_space=1, mlp_ratio=2.0, dropout=0.0,
                 pooling_method="cls", pos_embedding_method="learned",
                 use_causal_mask_time=True, use_causal_mask_space=False)
    return {
        "paths": {"raw_dir": str(raw), "save_dir": str(tmp_path / "out")},
        "data": {"split_test_size": 0.5, "seed": 42},
        "model": {"name": "gvt", **small},
        "training": {"batch_size": 16, "epochs": 2, "learning_rate": 1e-3},
        "kalman": {"q": 0.012, "r": 1.0, "method": "smooth"},
    }


def test_individual_training_and_evaluation(tmp_path):
    from emgpred.training.common import run_patients
    from emgpred.training.individual import train_patient

    raw = _fake_raw(tmp_path)
    cfg = _cfg(tmp_path, raw)
    run_patients(train_patient, ["AAA", "BBB"], cfg["paths"]["save_dir"], cfg=cfg)
    summary = pd.read_csv(tmp_path / "out" / "resumo_todos_pacientes.csv")
    assert list(summary["patient_id"]) == ["AAA", "BBB"]

    df = collect_results(cfg["paths"]["save_dir"], affine_calib_frac=0.5)
    assert {"raw_r2", "kalman_r2", "affine_r2"} <= set(df.columns)


def test_autopruning_training(tmp_path):
    from emgpred.training.autopruning import train_patient

    raw = _fake_raw(tmp_path)
    cfg = _cfg(tmp_path, raw)
    cfg["paths"]["saliency_dir"] = str(_fake_saliency(tmp_path))
    cfg["model"].pop("name")
    cfg["training"]["epochs"] = 4
    cfg["autopruning"] = {"k_final": 4, "warmup_epochs": 1, "prune_epochs": 1, "lambda_task": 0.1,
                          "tau_start": 2.0, "tau_end": 0.1, "calib_max": 100}
    res = train_patient("AAA", cfg)
    assert {"dyn_r2_raw", "pat_r2_raw", "pattask_r2_raw"} <= set(res)
    sel = json.load(open(tmp_path / "out" / "paciente_AAA" / "eletrodos_selecionados.json"))
    assert all(len(v) == 4 for v in sel.values())


def test_cli_train_individual_with_selection(tmp_path):
    from emgpred.cli import main

    raw = _fake_raw(tmp_path)
    sal = _fake_saliency(tmp_path)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f"""
[paths]
raw_dir = "{raw}"
saliency_dir = "{sal}"
save_dir = "{tmp_path}/fixo_{{selection}}"
[data]
selection = "todos"
k = 4
split_test_size = 0.5
[model]
name = "gvt"
d_model = 16
n_heads = 2
n_layers_time = 1
n_layers_space = 1
mlp_ratio = 2.0
dropout = 0.0
[training]
batch_size = 16
epochs = 1
""")
    main(["train", "individual", "-c", str(cfg), "-p", "AAA", "--set", "data.selection=media"])
    summary = pd.read_csv(tmp_path / "fixo_media" / "paciente_AAA" / "results_summary.csv")
    assert summary.loc[0, "electrodes"] == "8 9 10 11"       # 4 de maior ImportanciaMediaGlobal


def test_npy_cache_matches_npz(tmp_path):
    from emgpred.data.datasets import load_split

    raw = _fake_raw(tmp_path)
    a = load_split(str(raw), "AAA", "train", channels=[3, 1], with_task=True)
    b = load_split(str(raw), "AAA", "train", channels=[3, 1], with_task=True, cache_dir=str(tmp_path / "cache"))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_loso_fold_and_training(tmp_path):
    from emgpred.data.datasets import load_loso_fold, validation_patients
    from emgpred.training.loso import train_patient

    raw = _fake_raw(tmp_path, pids=("AAA", "BBB", "CCC", "DDD"))
    assert validation_patients(["AAA", "BBB", "CCC", "DDD"], "CCC", 2) == ["DDD", "AAA"]   # circular

    # sorteio na pool (_train + _test)
    (X, Y, tr), (_, _, va), vps = load_loso_fold(str(raw), "AAA", channels=[0, 5], val_patients=0,
                                                 val_fraction=0.1, pool_test=True)
    assert X.shape == (3 * (96 + 48), 20, 2) and len(tr) + len(va) == len(X) and not set(tr) & set(va) and not vps

    # validação por paciente, pool só com _train e padronização por paciente
    (X, Y, tr), (X_va, _, _), vps = load_loso_fold(str(raw), "AAA", val_patients=1, normalize=True)
    assert vps == ["BBB"] and len(X) == 2 * 96 and len(X_va) == 48
    assert abs(float(X.astype(np.float32).mean())) < 0.1                 # padronizado

    cfg = _cfg(tmp_path, raw)
    cfg["data"] = {"selection": "todos", "val_patients": 1, "normalize": True, "pool_test": False}
    cfg["training"].update(steps_per_epoch=3, epochs=3, val_smooth=2)
    res = train_patient("AAA", cfg)
    assert res["train_samples"] == 2 * 96 and res["test_samples"] == 48 and res["val_patients"] == "BBB"
    assert res["seen_n"] == 2                                             # CCC e DDD avaliados como vistos


def test_temporal_split_and_aug_search(tmp_path):
    from emgpred.data.datasets import temporal_split
    from emgpred.training.aug_search import AugSearch

    rec = np.repeat([0, 1], 50)
    tr, va = temporal_split(rec, val_fraction=0.2, gap_windows=4)
    assert list(va[:10]) == list(range(40, 50)) and tr.max() < 90 and 36 not in tr and 35 in tr

    raw = _fake_raw(tmp_path)
    for f in raw.glob("*_train.npz"):                                    # 2 "gravações" por paciente
        d = dict(np.load(f, allow_pickle=True))
        meta = d["metadata"].item()
        meta["filename"] = ["rec0"] * 48 + ["rec1"] * 48
        np.savez_compressed(f, emg=d["emg"], angles=d["angles"], metadata=meta)
    cfg = _cfg(tmp_path, raw)
    cfg["paths"]["search_dir"] = str(tmp_path / "busca")
    cfg["data"] = {"selection": "todos", "seed": 42}
    cfg["training"]["epochs"] = 1
    cfg["search"] = {"method": "greedy", "tune_patients": ["AAA"], "candidates": ["amplitude", "ruido"],
                     "min_rel_gain": -10.0}                              # aceita qualquer camada
    chosen = AugSearch(cfg).search()
    assert chosen == ["amplitude", "ruido"]
    import pandas as pd
    assert len(pd.read_csv(tmp_path / "busca" / "trials.csv")) == 4      # nenhuma, +amp, +ruido, +amp+ruido


def test_random_selection_and_holm():
    from emgpred.evaluation.saliency import electrode_selection
    from emgpred.evaluation.stats import holm

    a = electrode_selection("aleatorio", random_seed=0)
    assert a == sorted(int(c) for c in np.random.default_rng(0).choice(128, 32, replace=False)) and len(set(a)) == 32
    assert a != electrode_selection("aleatorio", random_seed=1)
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
