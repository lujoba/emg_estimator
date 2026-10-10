"""Callbacks de treino."""
import importlib.util
import os

import keras
import numpy as np


def _set_lr(model, lr):
    model.optimizer.learning_rate.assign(lr)


class CyclicLRScheduler(keras.callbacks.Callback):
    """LR triangular entre base_lr e max_lr, com meio-ciclo de `step_size` batches."""

    def __init__(self, base_lr=1e-5, max_lr=1e-3, step_size=2000, verbose=0):
        super().__init__()
        self.base_lr, self.max_lr, self.step_size, self.verbose = base_lr, max_lr, step_size, verbose
        self.step_count = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.step_count += 1
        cycle = np.floor(1 + self.step_count / (2 * self.step_size))
        x = np.abs(self.step_count / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0.0, 1 - x)
        _set_lr(self.model, lr)
        if self.verbose and self.step_count % 100 == 0:
            print(f"Step {self.step_count}: LR = {lr:.7f}")


class WarmUpScheduler(keras.callbacks.Callback):
    """Rampa linear do LR até `initial_lr` nos primeiros `warmup_steps` batches."""

    def __init__(self, warmup_steps, initial_lr, verbose=0):
        super().__init__()
        self.warmup_steps, self.initial_lr, self.verbose = warmup_steps, initial_lr, verbose
        self.step_count = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            _set_lr(self.model, self.initial_lr * self.step_count / self.warmup_steps)


class PruneScheduler(keras.callbacks.Callback):
    """k: n_ch (warmup) -> k_final (cúbico, Zhu & Gupta 2017); tau com decaimento exponencial."""

    def __init__(self, selector, k_final, warmup_steps, prune_steps, tau_start, tau_end):
        super().__init__()
        self.selector = selector
        self.k_final = k_final
        self.warmup_steps, self.prune_steps = warmup_steps, prune_steps
        self.tau_start, self.tau_end = tau_start, tau_end
        self.step = 0
        self._last = None

    @property
    def done(self):
        return self.step >= self.warmup_steps + self.prune_steps

    def values(self, step):
        n = self.selector.n_ch
        frac = np.clip((step - self.warmup_steps) / max(1, self.prune_steps), 0.0, 1.0)
        k = int(round(self.k_final + (n - self.k_final) * (1.0 - frac) ** 3))
        frac_tau = min(1.0, step / max(1, self.warmup_steps + self.prune_steps))
        tau = self.tau_start * (self.tau_end / self.tau_start) ** frac_tau
        return k, round(float(tau), 4)

    def on_train_batch_begin(self, batch, logs=None):
        values = self.values(self.step)
        if values != self._last:
            self.selector.k.assign(float(values[0]))
            self.selector.tau.assign(values[1])
            self._last = values
        self.step += 1

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["k"] = float(self.selector.k.numpy())
            logs["tau"] = float(self.selector.tau.numpy())


class PostPruneTraining(keras.callbacks.Callback):
    """Checkpoint + ReduceLROnPlateau + EarlyStopping ativos só depois do pruning
    (com mais eletrodos a val_loss é menor e não é comparável).

    Mesmas regras dos callbacks do Keras usados nos modelos fixos: o ReduceLR só conta melhora maior
    que `lr_min_delta` (padrão do Keras, 1e-4); checkpoint e EarlyStopping aceitam qualquer melhora.
    """

    def __init__(self, scheduler, weights_path, monitor="val_angles_loss",
                 lr_patience=3, lr_factor=0.5, min_lr=1e-7, stop_patience=8, lr_min_delta=1e-4):
        super().__init__()
        self.scheduler = scheduler
        self.weights_path = weights_path
        self.monitor = monitor
        self.lr_patience, self.lr_factor, self.min_lr, self.lr_min_delta = lr_patience, lr_factor, min_lr, lr_min_delta
        self.stop_patience = stop_patience
        self.best, self.best_epoch, self.lr_best = np.inf, None, np.inf
        self.wait = self.lr_wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if not self.scheduler.done:
            return
        current = logs.get(self.monitor, logs.get("val_loss"))
        if current < self.lr_best - self.lr_min_delta:
            self.lr_best, self.lr_wait = current, 0
        else:
            self.lr_wait += 1
        if current < self.best:
            self.best, self.best_epoch, self.wait = current, epoch, 0
            self.model.save_weights(self.weights_path)
            print(f"\n   💾 melhor {self.monitor}={current:.5f} (época {epoch + 1})")
        else:
            self.wait += 1
        if self.lr_wait >= self.lr_patience:
            lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            new_lr = max(lr * self.lr_factor, self.min_lr)
            _set_lr(self.model, new_lr)
            print(f"\n   ↘ LR {lr:.2e} → {new_lr:.2e}")
            self.lr_wait = 0
        if self.wait >= self.stop_patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)


class SmoothedValLoss(keras.callbacks.Callback):
    """Acrescenta logs['val_loss_smooth'] = média móvel (em épocas) da val_loss; os callbacks seguintes
    podem monitorar essa chave (PhD_LOSO, VAL_SMOOTH)."""

    def __init__(self, window):
        super().__init__()
        self.window, self.values = window, []

    def on_epoch_end(self, epoch, logs=None):
        self.values.append(logs["val_loss"])
        logs["val_loss_smooth"] = float(np.mean(self.values[-self.window:]))


def standard_callbacks(out_dir, checkpoint_name="model.weights.h5", save_weights_only=True,
                       early_stop_patience=15, lr_patience=10, lr_factor=0.75, min_lr=1e-8,
                       tensorboard=True, collapse_guard=True, val_smooth=1):
    """CollapseGuard + ModelCheckpoint + EarlyStopping + ReduceLROnPlateau (+ TensorBoard), em val_loss.

    O CollapseGuard (se ativo) é o primeiro da lista; recupere-o com `cbs[0]` para ler `rollbacks`.
    Com `val_smooth` > 1, checkpoint, EarlyStopping e ReduceLR monitoram a média móvel da val_loss.
    """
    monitor = "val_loss_smooth" if val_smooth > 1 else "val_loss"
    cbs = [CollapseGuard(monitor="val_loss")] if collapse_guard else []
    if val_smooth > 1:
        cbs.append(SmoothedValLoss(val_smooth))
    cbs += [
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, checkpoint_name), monitor=monitor,
                                        save_best_only=True, save_weights_only=save_weights_only, mode="min"),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=early_stop_patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="min", factor=lr_factor, patience=lr_patience,
                                          min_lr=min_lr),
        keras.callbacks.CSVLogger(os.path.join(out_dir, "training_history.csv")),
    ]
    if tensorboard and importlib.util.find_spec("tensorboard") is not None:
        cbs.append(keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir, "tensorboard_logs")))
    return cbs


class CollapseGuard(keras.callbacks.Callback):
    """Se a perda de validação explodir (> factor × a melhor), volta aos melhores pesos e corta o LR.

    Às vezes o transformer diverge e passa a prever uma constante (R² ≈ 0) sem sair dali.
    `on_restore` é chamado depois da restauração (o autopruning usa para reatribuir k e τ).
    """

    def __init__(self, monitor="val_loss", factor=3.0, lr_factor=0.5, on_restore=None):
        super().__init__()
        self.monitor, self.factor, self.lr_factor, self.on_restore = monitor, factor, lr_factor, on_restore
        self.best, self.best_weights, self.rollbacks = np.inf, None, 0

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        if current < self.best:
            self.best, self.best_weights = current, self.model.get_weights()
        elif self.best_weights is not None and current > self.factor * self.best:
            self.model.set_weights(self.best_weights)
            lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            _set_lr(self.model, lr * self.lr_factor)
            self.rollbacks += 1
            if self.on_restore is not None:
                self.on_restore()
            print(f"\n   ⚠️ colapso ({self.monitor}={current:.4f} > {self.factor:g}× melhor {self.best:.4f}): "
                  f"pesos restaurados, LR {lr:.1e} → {lr * self.lr_factor:.1e}")
