"""Figuras reutilizadas pelos scripts e notebooks."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from emgpred.constants import FINGERS, TASKS


def plot_predictions(y_true, y_pred, title="", path=None, finger_names=FINGERS):
    """Ground truth vs predição, um painel por dedo."""
    fig, axes = plt.subplots(y_true.shape[1], 1, figsize=(24, 4 * y_true.shape[1]), sharex=True)
    fig.suptitle(title, fontsize=14)
    for j, ax in enumerate(axes):
        r2_j = r2_score(y_true[:, j], y_pred[:, j])
        mae_j = np.mean(np.abs(y_true[:, j] - y_pred[:, j]))
        ax.plot(y_true[:, j], label="Original")
        ax.plot(y_pred[:, j], label="Predito")
        ax.set_title(f"{finger_names[j]}  R²={r2_j:.4f}  MAE={mae_j:.4f}")
        ax.set_ylabel("Ângulo (normalizado)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Amostras")
    fig.tight_layout()
    return _finish(fig, path)


def plot_electrode_selection(freq_task, masks, fixed_sets, path=None):
    """Frequência de seleção por tarefa e máscaras fixas (autopruning).

    fixed_sets: {nome: lista de eletrodos} das seleções globais para comparação.
    """
    n_ch = freq_task.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(22, 8), gridspec_kw={"height_ratios": [len(TASKS), len(TASKS) + 2]})
    im = axes[0].imshow(freq_task, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Frequência de seleção dinâmica no teste (por tarefa real)")
    axes[0].set_yticks(range(len(TASKS)), TASKS)
    fig.colorbar(im, ax=axes[0], fraction=0.02)
    fixed = np.zeros((len(fixed_sets), n_ch))
    for r, sel in enumerate(fixed_sets.values()):
        fixed[r, sel] = 1
    rows = np.vstack([masks[:-1], masks[-1], fixed])
    axes[1].imshow(rows, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("Máscaras fixas: por tarefa predita, paciente e seleções globais por saliência")
    axes[1].set_yticks(range(len(rows)), TASKS + ["paciente"] + [f"global ({n})" for n in fixed_sets])
    axes[1].set_xlabel("Eletrodo")
    fig.tight_layout()
    return _finish(fig, path)


def plot_paired_boxplot(df, columns, labels=None, ylabel="", path=None):
    """Boxplot por modelo com uma linha cinza ligando o mesmo paciente."""
    vals = df[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(max(6, 2.4 * len(columns)), 5))
    for row in vals:
        ax.plot(range(len(columns)), row, color="0.8", lw=0.8, zorder=1)
    ax.boxplot(vals, positions=range(len(columns)), widths=0.5, showfliers=False)
    ax.set_xticks(range(len(columns)), labels or columns, rotation=15)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return _finish(fig, path)


def plot_efficient_frontier(result, path=None):
    """Nuvem Monte Carlo (risco × retorno), fronteira e subconjunto escolhido."""
    r, s, f, b = result["returns"], result["risks"], result["frontier"], result["best"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s, r, alpha=0.2, s=10, label="Subconjuntos (Monte Carlo)")
    ax.scatter(s[f], r[f], color="red", s=20, label="Fronteira eficiente")
    ax.scatter(s[b], r[b], color="black", s=150, marker="*", zorder=5, label="Escolhido")
    ax.set_xlabel("Risco (desvio da saliência)")
    ax.set_ylabel("Retorno (saliência média)")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, path)


def _finish(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_electrode_heatmap(global_table, k=32, markowitz=None, path=None,
                           titles=("Electrode Array - FDS", "Electrode Array - EDC")):
    """Saliência média normalizada (± desvio) nas duas grades 8×8, como no `PhD_GlobalAvgHeatmapElectrodes`.

    global_table: DataFrame com `ImportanciaMediaGlobal` e `VarianciaGlobal` (tabela_importancia_global.csv).
    Contorno: os `k` de maior saliência média; ponto: o subconjunto `markowitz`.
    Cada grade tem 64 canais em ordem de linha (canal = 64·grade + 8·(linha−1) + (coluna−1)), linha 1 embaixo.
    """
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    import seaborn as sns

    mean = global_table["ImportanciaMediaGlobal"].to_numpy()
    norm = mean / mean.max()
    sd = np.sqrt(global_table["VarianciaGlobal"].to_numpy()) / mean.max()
    annot = np.array([f"{m:.2f}±{s:.2f}" for m, s in zip(norm, sd)])
    top = set(np.argsort(-mean)[:k])
    marked = set(markowitz or [])
    base = plt.get_cmap("Blues")
    cmap = mcolors.LinearSegmentedColormap.from_list("blues_trunc", base(np.linspace(0.1, 0.75, 256)))
    sel_color = "darkred"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"wspace": 0.08})
    for g, (ax, title) in enumerate(zip(axes, titles)):
        block = norm[64 * g:64 * g + 64].reshape(8, 8)
        hmap = sns.heatmap(ax=ax, data=np.flipud(block), annot=np.flipud(annot[64 * g:64 * g + 64].reshape(8, 8)),
                           fmt="", cmap=cmap, linewidths=.5, cbar=False, vmin=norm.min(), vmax=norm.max(),
                           annot_kws={"color": "black", "size": 6})
        for i in range(8):
            for j in range(8):
                ch = 64 * g + (7 - i) * 8 + j
                if ch in top:
                    ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, edgecolor=sel_color, lw=3))
                    for t in ax.texts:
                        if np.isclose(t.get_position()[0], j + 0.5) and np.isclose(t.get_position()[1], i + 0.5):
                            t.set_color(sel_color)
                            break
                if ch in marked:
                    ax.plot(j + 0.86, i + 0.16, marker="o", ms=3.5, color="black")
        ax.set_title(title)
        ax.set_xlabel("Channel Column", fontsize=12)
        ax.set_xticks(np.arange(8) + 0.5)
        ax.set_xticklabels(np.arange(1, 9))
        if g == 0:
            ax.set_ylabel("Channel Row", fontsize=12)
            ax.set_yticks(np.arange(8) + 0.5)
            ax.set_yticklabels(np.arange(8, 0, -1))
        else:
            ax.set_yticks([])
    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
    cb = fig.colorbar(hmap.collections[0], cax=cbar_ax)
    cb.set_label("Normalized Importance", rotation=-90, va="bottom", labelpad=14, fontsize=12)
    handles = [mlines.Line2D([], [], color=sel_color, marker="s", linestyle="None", markersize=10,
                             markerfacecolor="none", markeredgewidth=1.6, label=f"Top-{k} (mean saliency)")]
    if marked:
        handles.append(mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=4,
                                     label="Markowitz subset"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=11, frameon=False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
    return _finish(fig, path)
