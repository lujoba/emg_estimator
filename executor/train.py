import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from einops.layers.tensorflow import Rearrange
from models.estimator import estimator
from models.data_augmentation import AmplitudeScalingLayer, TimeShiftLayer, PhaseRotationLayer
from matplotlib import pyplot as plt


# Definir os callbacks
checkpoint_callback = ModelCheckpoint(
    filepath="gdrive/MyDrive/Colab_Notebooks/estimator/logs/model_1DoF_checkpoint/estimator_1/",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)

tensorboard_callback = TensorBoard(
    log_dir='gdrive/MyDrive/Colab_Notebooks/estimator/logs/model_1DoF_tensorboard/estimator_1/',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
)

early_stopping_callback = EarlyStopping(
    patience=30,
    restore_best_weights=True
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.75,
    patience=5,
    min_lr=1e-8,
)

callbacks = [checkpoint_callback, tensorboard_callback, early_stopping_callback, reduce_lr_callback]

def train_model(dataset_train, dataset_val, model, callbacks=callbacks, batch_size=8):
    batch_size = batch_size
    # Compilar o modelo
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse')

    # Treinar o modelo
    history = model.fit(
        dataset_train,
        epochs=100,
        validation_data=dataset_val,
        callbacks=callbacks,
    )

    # Plot training & validation loss values
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title('Loss')
    ax.plot(history.history['loss'], label = 'train')
    ax.plot(history.history['val_loss'], label = 'test')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Validation'])
