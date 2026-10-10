"""Linha de comando: `emgpred <subcomando> [-c config.toml] [-p PID ...] [--set secao.chave=valor ...]`.

Subcomandos:
    dataset       CSV brutos (EMG + cinemática) -> <PID>_dataset_{train,test}.npz
    train         treino por paciente: individual | loso | autopruning
    importance    saliência por eletrodo/tarefa + seleção de k eletrodos (fronteira eficiente)
    evaluate      métricas das predições salvas, Wilcoxon pareado e não inferioridade
    latency       latência de inferência por configuração (eletrodos × d_model, fixo/dinâmico)
    aug-search    busca da combinação de data augmentation (validação temporal)
"""
import argparse
import os

from emgpred.config import load_config

DEFAULT_CONFIGS = {
    "dataset": "configs/dataset.toml",
    "individual": "configs/gvt_individual.toml",
    "loso": "configs/gvt_loso.toml",
    "autopruning": "configs/autopruning.toml",
    "importance": "configs/saliency.toml",
    "aug-search": "configs/aug_search.toml",
}


def _common(p, default_config):
    p.add_argument("-c", "--config", default=default_config, help=f"arquivo TOML (padrão: {default_config})")
    p.add_argument("-p", "--patients", nargs="*", help="IDs dos pacientes (padrão: todos em raw_dir)")
    p.add_argument("--set", nargs="*", default=[], metavar="SECAO.CHAVE=VALOR",
                   help="sobrescreve chaves do config, ex.: --set training.epochs=5 data.selection=sharpe")


def _patients(args, cfg):
    from emgpred.data.datasets import list_patients
    return [p.upper() for p in args.patients] if args.patients else list_patients(cfg["paths"]["raw_dir"])


# ---------------------------------------------------------------------------
def cmd_dataset(args):
    from emgpred.data.datasets import build_all_datasets
    cfg = load_config(args.config, args.set)
    build_all_datasets(cfg["paths"]["emg_root"], cfg["paths"]["angles_root"], cfg["paths"]["raw_dir"],
                       **cfg["dataset"])


def cmd_train(args):
    import importlib

    from emgpred.training.common import run_patients
    cfg = load_config(args.config or DEFAULT_CONFIGS[args.protocol], args.set)
    # save_dir/references podem usar campos de [data], {sel} (seleção; "aleatorio<seed>" no sorteio),
    # {suffix} ("" para d_model 128, senão "_d<N>"), {model}, {aug} ("" ou "_aug-com-<nomes>") e
    # {loso_tag} ("_vp4_ma3_norm_trainonly"...), ex.: "logs/fixo_{sel}_8020{suffix}{aug}"
    fields = _format_fields(cfg)
    cfg["paths"]["save_dir"] = cfg["paths"]["save_dir"].format(**fields)
    cfg["references"] = {k: v.format(**fields) for k, v in cfg.get("references", {}).items()}
    train_patient = importlib.import_module(f"emgpred.training.{args.protocol}").train_patient
    kwargs = {"eval_only": args.eval_only} if args.protocol == "autopruning" else {}
    print(f"▶ {args.protocol} → {cfg['paths']['save_dir']}")
    run_patients(train_patient, _patients(args, cfg), cfg["paths"]["save_dir"],
                 skip_done=not (args.no_skip or args.eval_only), cfg=cfg, **kwargs)


def _format_fields(cfg):
    """Campos usados nos nomes de pasta dos configs ({sel}, {suffix}, {aug}, {model}, {loso_tag}...)."""
    from emgpred.models.augmentation import augmentation_tag
    from emgpred.training.common import selection_tag
    data, d_model = cfg.get("data", {}), cfg.get("model", {}).get("d_model", 128)
    loso_tag = ((f"_vp{data['val_patients']}" if data.get("val_patients") else "")
                + (f"_ma{cfg['training']['val_smooth']}" if cfg.get("training", {}).get("val_smooth", 1) > 1 else "")
                + ("_norm" if data.get("normalize") else "") + ("_init" if cfg["paths"].get("init_from") else "")
                + ("" if data.get("pool_test", True) else "_trainonly"))
    return {**data, "sel": selection_tag(data), "loso_tag": loso_tag, "d_model": d_model,
            "suffix": "" if d_model == 128 else f"_d{d_model}", "model": cfg.get("model", {}).get("name", "gvt"),
            "aug": augmentation_tag(cfg.get("augmentation", {}).get("names", []))}


def cmd_aug_search(args):
    from emgpred.training.aug_search import AugSearch
    cfg = load_config(args.config, args.set)
    cfg["paths"]["search_dir"] = cfg["paths"]["search_dir"].format(**_format_fields(cfg))
    print(f"▶ busca de augmentation → {cfg['paths']['search_dir']}")
    AugSearch(cfg).search()


def cmd_importance(args):
    import numpy as np
    import pandas as pd

    from emgpred.data.datasets import load_split
    from emgpred.data.datasets import read_channel_list
    from emgpred.evaluation.plots import plot_efficient_frontier, plot_electrode_heatmap
    from emgpred.evaluation.saliency import efficient_frontier_selection, saliency_per_task, save_task_tables
    from emgpred.training.common import build_model

    cfg = load_config(args.config, args.set)
    paths, sel = cfg["paths"], cfg["selection"]
    os.makedirs(paths["save_dir"], exist_ok=True)
    importances = {}
    for pid in _patients(args, cfg):
        weights = os.path.join(paths["model_dir"], f"paciente_{pid}", paths["weights_name"])
        if not os.path.exists(weights):
            print(f"⚠️  pesos não encontrados para {pid}")
            continue
        X, Y, T = load_split(paths["raw_dir"], pid, "test", with_task=True)
        model = build_model(cfg["model"], X.shape[1], X.shape[2])
        model.load_weights(weights)
        importances[f"paciente_{pid}"] = saliency_per_task(model, X, Y, T)
        print(f"✅ {pid}")

    n_ch = next(iter(importances.values())).shape[1]
    save_task_tables(importances, paths["save_dir"], n_ch)
    matrix = np.concatenate([v[~np.isnan(v).any(axis=1)] for v in importances.values()])
    pd.DataFrame({"ImportanciaMediaGlobal": matrix.mean(0), "VarianciaGlobal": matrix.var(0)},
                 index=[f"E_{j}" for j in range(n_ch)]).to_csv(
        os.path.join(paths["save_dir"], "tabela_importancia_global.csv"))
    res = efficient_frontier_selection(matrix, k=sel["k"], n_samples=sel["n_samples"], seed=sel["seed"])
    pd.DataFrame({"EletrodosSelecionados": [f"E_{i}" for i in res["subset"]],
                  "ImportanciaMedia": res["mu"][res["subset"]]}).sort_values(
        "ImportanciaMedia", ascending=False).to_csv(
        os.path.join(paths["save_dir"], f"melhores_{sel['k']}_eletrodos.csv"), index=False)
    plot_efficient_frontier(res, os.path.join(paths["save_dir"], "fronteira_eficiente.png"))
    best_file = paths.get("best_channels_file", "configs/channels/best32.json")
    markowitz = read_channel_list(best_file) if os.path.exists(best_file) else res["subset"].tolist()
    plot_electrode_heatmap(pd.read_csv(os.path.join(paths["save_dir"], "tabela_importancia_global.csv"), index_col=0),
                           k=sel["k"], markowitz=markowitz,
                           path=os.path.join(paths["save_dir"], f"heatmap_top{sel['k']}_markowitz.pdf"))
    print(f"Selecionados: {res['subset'].tolist()}")


def cmd_evaluate(args):
    import pandas as pd

    from emgpred.evaluation.results import collect_results, summarize
    from emgpred.evaluation.stats import friedman, holm, non_inferiority, paired_comparison

    os.makedirs(args.out, exist_ok=True)
    kalman_kwargs = {"method": args.kalman_method, "q": args.kalman_q, "r": args.kalman_r}
    if args.kalman_method in ("window", "fixed_lag"):
        kalman_kwargs["half_window" if args.kalman_method == "window" else "lag"] = args.half_window
    tables = {}
    for spec in args.runs:
        name, _, rest = spec.partition("=")
        run_dir, _, mode = rest.partition("|")          # "DIR|pat" -> predictions_pat.csv (autopruning)
        print(f"\n=== {name} ({rest}) ===")
        df = collect_results(run_dir, kalman_kwargs=kalman_kwargs, affine_calib_frac=args.affine_calib,
                             ma_kwargs={"half_window": args.ma_half_window, "causal": args.ma_causal},
                             pred_name=f"predictions_{mode}.csv" if mode else None)
        df.to_csv(os.path.join(args.out, f"metricas_{name}.csv"), index=False)
        cols = [c for c in df.columns if c.split("_")[0] in ("raw", "kalman", "ma", "affine") and c.count("_") == 1]
        print(summarize(df, cols).round(4).to_string())
        tables[name] = df

    s = args.stage
    pick = lambda df: df[["patient", f"{s}_r2", f"{s}_mae"]].rename(columns=lambda c: c.replace(f"{s}_", ""))
    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else \
        [(a, b) for i, a in enumerate(tables) for b in list(tables)[i + 1:]]
    if len(tables) > 2:
        wide = pd.DataFrame({n: t.set_index("patient")[f"{s}_r2"] for n, t in tables.items()})
        fr = friedman(wide)
        print(f"\nFriedman (R² {s}, {fr['k']} modelos, n={fr['n']}): χ²={fr['chi2']:.3f} p={fr['p']:.2e}")
    rows = []
    for a, b in pairs:
        cmp = paired_comparison(pick(tables[a]), pick(tables[b]), labels=("A", "B"))
        rows.append(cmp.assign(A=a, B=b))
    if rows:
        cmp = pd.concat(rows, ignore_index=True)
        for metric in cmp["metric"].unique():          # Holm entre os pares, por métrica
            m = cmp["metric"] == metric
            cmp.loc[m, "p_holm"] = holm(cmp.loc[m, "p"].to_numpy())
        cols = ["A", "B", "metric", "n", "mean_A", "mean_B", "delta_mean", "n_A_greater", "p", "p_holm", "r_rb", "effect"]
        print(f"\nWilcoxon pareado ({s}), p ajustado por Holm:\n{cmp[cols].round(5).to_string(index=False)}")
        cmp[cols].to_csv(os.path.join(args.out, f"wilcoxon_{s}.csv"), index=False)

    rows = []
    for spec in args.noninf:
        a, b = spec.split(":")
        m = pick(tables[a]).merge(pick(tables[b]), on="patient", suffixes=("_a", "_b"))
        rows.append({"A": a, "B": b, **non_inferiority(m["r2_a"], m["r2_b"], margin=args.margin)})
    if rows:
        df = pd.DataFrame(rows)
        print(f"\nNão inferioridade (R² {s}, margem {args.margin}):\n{df.round(4).to_string(index=False)}")
        df.to_csv(os.path.join(args.out, f"nao_inferioridade_{s}.csv"), index=False)


def cmd_latency(args):
    import keras
    import numpy as np
    import tensorflow as tf

    from emgpred.evaluation.latency import benchmark
    from emgpred.models.gvt import build_gvt
    from emgpred.models.selector import build_autopruning_model

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    cfg = load_config(args.config, args.set)
    backbone = {k: v for k, v in cfg["model"].items() if k not in ("name", "d_model")}
    print("CPU, 1 thread" if args.cpu else tf.config.list_physical_devices("GPU"))
    configs = [(f"fixo {c} el., d{d}", c, d, False) for d in args.d_model for c in args.channels]
    if args.dynamic:
        configs += [(f"dinâmico 128->{args.k}, d{d}", 128, d, True) for d in args.d_model]
    for name, n_ch, d, dyn in configs:
        if dyn:
            prior = np.full((8, 128), 1 / 128, "float32")
            model, _, sel = build_autopruning_model(prior, n_ch=128, backbone={**backbone, "d_model": d})
            sel.k.assign(float(args.k))
        else:
            model = build_gvt(100, n_ch, 1, 5, d_model=d, **backbone)
        reps = 50 if args.cpu else 300
        line = f"{name:24s} | {model.count_params():>9,} parâm. | 1 janela {benchmark(model, 1, reps, xla=not args.cpu):7.2f} ms"
        if not args.cpu:
            line += f" | lote 128 {benchmark(model, 128, 50):7.2f} ms"
        print(line, flush=True)
        keras.backend.clear_session()


# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(prog="emgpred", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dataset", help="CSV -> NPZ por paciente")
    _common(p, DEFAULT_CONFIGS["dataset"])
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("train", help="treino por paciente")
    p.add_argument("protocol", choices=["individual", "loso", "autopruning"])
    _common(p, None)
    p.add_argument("--no-skip", action="store_true", help="retreina pacientes que já têm results_summary.csv")
    p.add_argument("--eval-only", action="store_true", help="(autopruning) só avalia com os pesos salvos")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("aug-search", help="busca da combinação de data augmentation")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIGS["aug-search"])
    p.add_argument("--set", nargs="*", default=[], metavar="SECAO.CHAVE=VALOR")
    p.set_defaults(func=cmd_aug_search)

    p = sub.add_parser("importance", help="saliência + seleção de eletrodos")
    _common(p, DEFAULT_CONFIGS["importance"])
    p.set_defaults(func=cmd_importance)

    p = sub.add_parser("evaluate", help="métricas + Wilcoxon a partir das predições salvas")
    p.add_argument("--runs", nargs="+", required=True, metavar="NOME=DIR[|modo]",
                   help="modo do autopruning (dyn/pat/pattask) lê predictions_<modo>.csv")
    p.add_argument("--pairs", nargs="*", metavar="A:B", help="pares do Wilcoxon (padrão: todos)")
    p.add_argument("--noninf", nargs="*", default=[], metavar="A:B", help="não inferioridade de A em relação a B")
    p.add_argument("--margin", type=float, default=0.01, help="margem de não inferioridade em R²")
    p.add_argument("--out", required=True)
    p.add_argument("--kalman-method", choices=["window", "fixed_lag", "filter", "smooth"], default="window")
    p.add_argument("--half-window", type=int, default=3, help="amostras de cada lado (window) ou atraso (fixed_lag)")
    p.add_argument("--kalman-q", type=float, default=0.012)
    p.add_argument("--kalman-r", type=float, default=1.0)
    p.add_argument("--affine-calib", type=float, default=None,
                   help="fração inicial do teste usada para ajustar a afim (avaliada só no restante)")
    p.add_argument("--ma-half-window", type=int, default=3, help="média móvel de 2h+1 predições")
    p.add_argument("--ma-causal", action="store_true", help="média móvel só com predições passadas")
    p.add_argument("--stage", default="kalman", choices=["raw", "kalman", "ma", "affine"],
                   help="estágio usado no Wilcoxon e na não inferioridade (ma = média móvel)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("latency", help="latência de inferência")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIGS["individual"], help="config com o backbone em [model]")
    p.add_argument("--set", nargs="*", default=[], metavar="SECAO.CHAVE=VALOR")
    p.add_argument("--channels", type=int, nargs="+", default=[128, 32])
    p.add_argument("--d-model", type=int, nargs="+", default=[128, 32])
    p.add_argument("--dynamic", action="store_true", help="inclui o modelo dinâmico (seletor + 128 entradas)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--cpu", action="store_true", help="CPU com 1 thread (sem XLA)")
    p.set_defaults(func=cmd_latency)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
