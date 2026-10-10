"""Seleção de eletrodos guiada por saliência (autopruning 128 -> k).

logits_c = alpha * log(p(tarefa|x)^T S)_c + delta_c(x)

- S (n_tarefas x n_eletrodos): saliência média por tarefa, sem o próprio paciente.
- p(tarefa|x): classificador leve de tarefa (perda auxiliar).
- delta(x): ajuste por janela, inicializado em zero.
- Top-k com straight-through: máscara dura no forward, sigmoide com temperatura no backward.
"""
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from emgpred.models.gvt import gvt_backbone


@keras.saving.register_keras_serializable(package="emgpred")
class SaliencyGatedSelector(layers.Layer):
    MODE_DYNAMIC, MODE_PATIENT, MODE_PATIENT_TASK = 0, 1, 2

    def __init__(self, saliency_prior, hidden=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.prior_init = np.asarray(saliency_prior, np.float32)
        self.n_tasks, self.n_ch = self.prior_init.shape
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.hidden_dense = layers.Dense(hidden, activation="gelu")
        self.drop = layers.Dropout(dropout)
        self.task_head = layers.Dense(self.n_tasks, name="task_head")
        self.delta_head = layers.Dense(self.n_ch, kernel_initializer="zeros", bias_initializer="zeros",
                                       name="delta_head")

    def build(self, input_shape):
        const = keras.initializers.Constant
        self.prior = self.add_weight(name="saliency_prior", shape=self.prior_init.shape,
                                     initializer="zeros", trainable=False)
        self.prior.assign(self.prior_init)
        self.alpha = self.add_weight(name="prior_scale", shape=(), initializer=const(1.0), trainable=True)
        # Estado controlado de fora (PruneScheduler / avaliação)
        self.k = self.add_weight(name="k", shape=(), initializer=const(float(self.n_ch)), trainable=False)
        self.tau = self.add_weight(name="tau", shape=(), initializer=const(1.0), trainable=False)
        self.mode = self.add_weight(name="mode", shape=(), initializer="zeros", trainable=False)
        # linhas 0..n_tasks-1: máscara fixa por tarefa; última linha: máscara fixa do paciente
        self.fixed_masks = self.add_weight(name="fixed_masks", shape=(self.n_tasks + 1, self.n_ch),
                                           initializer="ones", trainable=False)
        super().build(input_shape)

    def call(self, x, training=None):
        s = tf.cast(x[..., 0], tf.float32)                                   # (B, T, C)
        rms = tf.sqrt(tf.reduce_mean(tf.square(s), axis=1) + 1e-8)           # (B, C)
        wl = tf.reduce_mean(tf.abs(s[:, 1:] - s[:, :-1]), axis=1)            # waveform length
        feats = self.norm(tf.concat([tf.math.log(rms), tf.math.log(wl + 1e-8)], axis=-1))
        h = self.drop(self.hidden_dense(feats), training=training)

        task_logits = self.task_head(h)
        p_task = tf.nn.softmax(task_logits)
        prior_logits = tf.math.log(tf.matmul(p_task, self.prior) + 1e-8)
        prior_logits -= tf.reduce_mean(prior_logits, axis=-1, keepdims=True)
        logits = self.alpha * prior_logits + self.delta_head(h)

        k = tf.cast(tf.round(self.k), tf.int32)
        top = tf.math.top_k(logits, k=self.n_ch)                             # ordenado decrescente
        # máscara dos k maiores com forma fixa (posição no ranking < k), sem recompilar o XLA a cada
        # novo k. Diferente de `logits >= k-ésimo valor`, seleciona exatamente k mesmo com empates
        # (ex.: no início do treino, com prior uniforme e delta_head zerado).
        rank = tf.argsort(top.indices, axis=-1, stable=True)                 # posição de cada eletrodo
        hard = tf.cast(rank < k, logits.dtype)

        if training:
            kth = tf.gather(top.values, k - 1, axis=1)
            nxt = tf.gather(top.values, tf.minimum(k, self.n_ch - 1), axis=1)
            thr = tf.stop_gradient(0.5 * (kth + nxt))[:, None]
            soft = tf.sigmoid((logits - thr) / self.tau)
            soft = tf.where(k >= self.n_ch, tf.ones_like(soft), soft)
            mask = hard + soft - tf.stop_gradient(soft)
        else:
            m_task = tf.gather(self.fixed_masks, tf.argmax(p_task, axis=-1))
            m_patient = tf.broadcast_to(self.fixed_masks[-1], tf.shape(hard))
            mask = tf.where(self.mode > 1.5, m_task, tf.where(self.mode > 0.5, m_patient, hard))

        return x * tf.cast(mask, x.dtype)[:, None, :, None], task_logits, logits


def build_autopruning_model(saliency_prior, time_steps=100, n_ch=128, output_dim=5, backbone=None):
    """Retorna (modelo de treino, modelo do gate, camada seletora)."""
    inputs = keras.Input(shape=(time_steps, n_ch, 1), name="emg")
    selector = SaliencyGatedSelector(saliency_prior, name="selector")
    x_sel, task_logits, gate_logits = selector(inputs)
    angles = gvt_backbone(x_sel, time_steps, n_ch, output_dim, **(backbone or {}))
    model = keras.Model(inputs, {"angles": angles, "task": task_logits}, name="AutoPruningGVT")
    gate_model = keras.Model(inputs, {"task": task_logits, "gate": gate_logits}, name="gate")
    return model, gate_model, selector


def topk_mask(scores, k):
    """Máscara binária com os k maiores valores por linha."""
    scores = np.atleast_2d(scores)
    idx = np.argpartition(-scores, k - 1, axis=-1)[:, :k]
    m = np.zeros_like(scores, dtype=np.float32)
    np.put_along_axis(m, idx, 1.0, axis=-1)
    return m


def calibrate_fixed_masks(gate_model, X_calib, k, n_tasks, batch_size=512, min_windows=50):
    """Fixa k eletrodos por paciente e por (paciente, tarefa predita) a partir da frequência
    de seleção dinâmica em janelas de calibração (sem rótulo)."""
    from emgpred.data.datasets import make_dataset

    out = gate_model.predict(make_dataset(X_calib, batch_size=batch_size), verbose=0)
    sel = topk_mask(out["gate"], k)
    task_pred = out["task"].argmax(-1)
    masks = np.zeros((n_tasks + 1, sel.shape[1]), np.float32)
    masks[-1] = topk_mask(sel.mean(0), k)[0]
    for t in range(n_tasks):
        in_t = task_pred == t
        masks[t] = topk_mask(sel[in_t].mean(0), k)[0] if in_t.sum() >= min_windows else masks[-1]
    return masks


def predict_angles(model, selector, ds, mode, fixed_masks=None):
    """Predição com o seletor no modo dinâmico (0), paciente (1) ou paciente+tarefa (2)."""
    if fixed_masks is not None:
        selector.fixed_masks.assign(fixed_masks)
    selector.mode.assign(float(mode))
    try:
        return model.predict(ds, verbose=0)["angles"]
    finally:
        selector.mode.assign(0.0)
