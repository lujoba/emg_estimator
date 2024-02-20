import tensorflow as tf


# Definindo os callbacks para salvar o modelo e o TensorBoard
def get_callbacks(model_name, subject_index):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"logs/models/{model_name}/na_all_fingers/subject_{subject_index}/",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_format="tf"
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'logs/models/{model_name}/na_all_fingers/subject_{subject_index}/',
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0,
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=20,
        restore_best_weights=True
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.75,
        patience=5,
        min_lr=1e-8,
    )

    return [model_checkpoint_callback, tensorboard_callback, early_stopping_callback, reduce_lr_callback]