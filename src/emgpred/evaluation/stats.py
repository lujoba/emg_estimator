"""Testes estatísticos pareados por paciente e tamanhos de efeito."""
import numpy as np
import pandas as pd
from scipy import stats


def rank_biserial_wilcoxon(stat, n):
    """r de correlação rank-biserial a partir da estatística W do Wilcoxon pareado."""
    return 1 - (2 * stat) / (n * (n + 1) / 2) if n else np.nan


def rank_biserial_mannwhitney(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)


def interpret_effect(r):
    r = abs(r)
    return "desprezível" if r < 0.1 else "pequeno" if r < 0.3 else "médio" if r < 0.5 else "grande"


def cohens_d_independent(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled


def paired_comparison(df_a, df_b, metrics=("r2", "mae"), on="patient", labels=("A", "B")):
    """Wilcoxon pareado entre dois modelos, só com os pacientes em comum.

    Retorna DataFrame com Δ médio (A − B), nº de pacientes em que A > B, p e r_rb.
    """
    merged = df_a.merge(df_b, on=on, suffixes=("_a", "_b"))
    rows = []
    for m in metrics:
        a, b = merged[f"{m}_a"].to_numpy(), merged[f"{m}_b"].to_numpy()
        res = stats.wilcoxon(a, b)
        n = int(np.sum(a != b))  # Wilcoxon descarta empates
        rows.append({
            "metric": m, "n": len(merged), f"mean_{labels[0]}": a.mean(), f"mean_{labels[1]}": b.mean(),
            "delta_mean": (a - b).mean(), f"n_{labels[0]}_greater": int((a > b).sum()),
            "W": res.statistic, "p": res.pvalue,
            "r_rb": rank_biserial_wilcoxon(res.statistic, n),
        })
    out = pd.DataFrame(rows)
    out["effect"] = out["r_rb"].map(interpret_effect)
    return out


def non_inferiority(a, b, margin=0.01, n_boot=10000, alpha=0.05, seed=0, min_n=6):
    """Teste de não inferioridade pareado: H0 = A perde de B mais que `margin` (Δ = A − B < −margin).

    Wilcoxon unilateral em (Δ + margin) > 0 e IC 95% bootstrap da Δ média. 'não inferior' exige
    p < alpha e o limite inferior do IC acima de −margin.
    """
    d = np.asarray(a) - np.asarray(b)
    if len(d) < min_n:   # com menos de 6 pares o Wilcoxon não chega a p < 0,05
        return {"n": len(d), "delta_mean": d.mean() if len(d) else np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p": np.nan, "margin": margin, "verdict": f"n < {min_n}"}
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = stats.wilcoxon(d + margin, alternative="greater").pvalue
    return {"n": len(d), "delta_mean": d.mean(), "ci_low": lo, "ci_high": hi, "p": p, "margin": margin,
            "verdict": "não inferior" if (p < alpha and lo > -margin) else "inconclusivo"}


def holm(pvalues):
    """Correção de Holm–Bonferroni para comparações múltiplas -> p ajustados (mesma ordem)."""
    p = np.asarray(pvalues, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def friedman(table):
    """Friedman entre modelos (colunas) com os mesmos pacientes (linhas sem NaN) -> dict."""
    t = table.dropna()
    res = stats.friedmanchisquare(*[t[c] for c in t.columns])
    return {"n": len(t), "k": t.shape[1], "chi2": res.statistic, "p": res.pvalue, "models": list(t.columns)}
