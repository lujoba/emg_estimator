import tensorflow as tf


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, model_name='model'):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.model = model_name

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            self.model.save_weights(self.filepath + f'{self.model_name}.h5', overwrite=True)
            print(f"\nSaving model weights at epoch {epoch + 1} with validation loss {val_loss:.4f}\n")


# Definindo os callbacks para salvar o modelo e o TensorBoard
def get_callbacks(model_name, subject_index, DA=False):
    """
    This function creates and returns a list of callback objects for training a machine-learning model.
    
    Args:
        - model_name (str): Name of the model being trained.
        - subject_index (int): Index of the subject the model is trained on.
        - DA (bool, optional): Flag indicating whether to use data augmentation (DA). Defaults to False.
    
    Returns:
        list: A list containing the following callback objects:
            - **model_checkpoint_callback:** Saves the model with the best validation loss.
            - **tensorboard_callback:** Logs training and validation metrics to TensorBoard.
            - **early_stopping_callback:** Stops training if validation loss doesn't improve for a certain number of epochs.
            - **reduce_lr_callback:** Reduces learning rate if validation loss doesn't improve for a certain number of epochs.
    
    Callback File Paths:
    
    - The callback objects save files to the following directory structure:
        - `logs/models/{model_name}/{DA_folder}/subject_{subject_index}/`
        - `DA_folder` is either "wa_all_fingers" (with DA) or "na_all_fingers" (no DA)
    """
    if DA:
        file_path=f'logs/models/{model_name}/wa_all_fingers/subject_{subject_index}/'
    else:
        file_path=f'logs/models/{model_name}/na_all_fingers/subject_{subject_index}/'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=file_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_format="tf"
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=file_path,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
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
        patience=10,
        min_lr=1e-8,
    )

    return [model_checkpoint_callback, tensorboard_callback, early_stopping_callback, reduce_lr_callback]