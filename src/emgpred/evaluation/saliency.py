"""Importância de eletrodos por mapas de saliência e seleção de subconjunto (fronteira eficiente).

Corresponde ao notebook `PhD_EletrodosMaisImportantes`.
"""
import os

import numpy as np
import pandas as pd

from emgpred.constants import TASKS


def saliency_per_task(model, X, y, task, n_tasks=len(TASKS), batch_size=256):
    """Saliência |∂MSE/∂x| média por eletrodo, separada por tarefa.

    Usa o rótulo de tarefa de cada janela (`task`, 0..n_tasks-1). O notebook original
    dividia o teste em 8 blocos de mesmo tamanho, assumindo que as tarefas estavam
    ordenadas e balanceadas; aqui o agrupamento usa o metadado.
    Retorna (n_tasks, n_eletrodos); linhas de tarefas ausentes ficam NaN.
    """
    import keras
    import tensorflow as tf

    n_ch = X.shape[2]
    out = np.full((n_tasks, n_ch), np.nan)
    for t in range(n_tasks):
        idx = np.flatnonzero(task == t)
        if len(idx) == 0:
            continue
        acc = np.zeros(n_ch)
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            x = tf.convert_to_tensor(X[b], dtype=tf.float32)
            if x.shape.rank == 3:
                x = x[..., None]
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = keras.losses.mean_squared_error(y[b], model(x, training=False))
            grad = tape.gradient(loss, x)
            acc += tf.reduce_sum(tf.reduce_mean(tf.abs(grad), axis=[1, 3]), axis=0).numpy()
        out[t] = acc / len(idx)
    return out


def save_task_tables(importances, save_dir, n_ch):
    """{paciente: (n_tasks, n_ch)} -> `tabela_importancia_movimento_<tarefa>.csv` (linhas = pacientes)."""
    os.makedirs(save_dir, exist_ok=True)
    cols = [f"E_{j}" for j in range(n_ch)]
    for t, name in enumerate(TASKS):
        rows = {p: imp[t] for p, imp in importances.items() if not np.isnan(imp[t]).all()}
        pd.DataFrame.from_dict(rows, orient="index", columns=cols).to_csv(
            os.path.join(save_dir, f"tabela_importancia_movimento_{name}.csv"))


def load_task_tables(sal_dir):
    """-> {tarefa: DataFrame (pacientes x E_0..E_{n-1})}."""
    return {name: pd.read_csv(os.path.join(sal_dir, f"tabela_importancia_movimento_{name}.csv"), index_col=0)
            for name in TASKS}


def saliency_prior(sal_dir, exclude_patient=None):
    """Prior (n_tasks, n_ch) para o autopruning: cada vetor paciente×tarefa é normalizado
    para somar 1 e depois promediado entre os pacientes (sem `exclude_patient`)."""
    prior = []
    for name, df in load_task_tables(sal_dir).items():
        if exclude_patient is not None:
            df = df.drop(index=f"paciente_{exclude_patient}", errors="ignore")
        S = df[[c for c in df.columns if c.startswith("E_")]].to_numpy(np.float64)
        prior.append((S / S.sum(axis=1, keepdims=True)).mean(axis=0))
    prior = np.stack(prior)
    return (prior / prior.sum(axis=1, keepdims=True)).astype(np.float32)


def efficient_frontier_selection(importance_matrix, k=32, n_samples=5000, seed=42):
    """Seleção de k eletrodos por analogia com carteira de Markowitz.

    Cada linha de `importance_matrix` é um vetor de saliência (paciente×tarefa).
    Sorteia `n_samples` subconjuntos de k eletrodos com peso uniforme, calcula
    retorno (média) e risco (desvio pela covariância) e escolhe, entre os
    não-dominados, o de maior razão retorno/risco.

    Retorna dict com subset escolhido, retornos, riscos e índices da fronteira.
    """
    # Os notebooks originais sorteavam sem semente, por isso cada execução escolhia um
    # subconjunto diferente (`melhores_32_eletrodos.csv` × `configs/channels/best32.json`).
    rng = np.random.default_rng(seed)
    mu = importance_matrix.mean(axis=0)
    sigma = np.cov(importance_matrix, rowvar=False)
    M = len(mu)

    subsets = np.stack([rng.choice(M, k, replace=False) for _ in range(n_samples)])
    W = np.zeros((n_samples, M))
    np.put_along_axis(W, subsets, 1.0 / k, axis=1)
    returns = W @ mu
    risks = np.sqrt(np.einsum("ij,jk,ik->i", W, sigma, W))

    dominated = ((returns[None, :] >= returns[:, None]) & (risks[None, :] <= risks[:, None])
                 & ((returns[None, :] > returns[:, None]) | (risks[None, :] < risks[:, None]))).any(axis=1)
    frontier = np.flatnonzero(~dominated)
    best = frontier[np.argmax(returns[frontier] / risks[frontier])]
    return {"subset": np.sort(subsets[best]), "mu": mu, "returns": returns, "risks": risks,
            "frontier": frontier, "best": best}



SELECTIONS = ("todos", "media", "sharpe", "best", "aleatorio")


def electrode_selection(name, saliency_dir=None, k=32, best_file=None, random_seed=0, n_channels=128):
    """Eletrodos de uma seleção fixa, ou None para todos.

    'media' : os k de maior `ImportanciaMediaGlobal` (`tabela_importancia_global.csv`), ordenados.
    'sharpe': subconjunto da fronteira eficiente (`melhores_<k>_eletrodos.csv`), ordenados.
    'best'  : lista `best_file` (subconjunto Markowitz usado nos treinos), ordenada como nos notebooks.
    'aleatorio': k eletrodos sorteados com `random_seed` (controle de acaso, PhD_SelecaoAleatoria).
    """
    if name == "todos":
        return None
    if name == "media":
        df = pd.read_csv(os.path.join(saliency_dir, "tabela_importancia_global.csv"), index_col=0)
        return sorted(int(e.split("_")[1]) for e in df["ImportanciaMediaGlobal"].nlargest(k).index)
    if name == "sharpe":
        df = pd.read_csv(os.path.join(saliency_dir, f"melhores_{k}_eletrodos.csv"))
        return sorted(int(e.split("_")[1]) for e in df["EletrodosSelecionados"])
    if name == "best":
        from emgpred.data.datasets import read_channel_list
        return sorted(read_channel_list(best_file))
    if name == "aleatorio":
        return sorted(int(c) for c in np.random.default_rng(random_seed).choice(n_channels, k, replace=False))
    raise ValueError(f"seleção desconhecida: {name} (opções: {SELECTIONS})")
