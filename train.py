import os
import tensorflow as tf
import keras
import importlib
import argparse

from utils.utils import *
from utils.data_lists import *
from utils.callbacks import get_callbacks
from dataloader.dataloader import *
from models.data_augmentation import *


def main(args):
    """
    Main function to train the model with the HYPER dataset.

    Args:
        args: argparse arguments containing configuration options and the model name.

    Raises:
        ValueError: If the model name is not found or arguments are invalid.
    """
    # Importing model module
    try:
        # Load configuration
        config = load_config(args.config)

        # Import model module
        try:
            model_module = importlib.import_module(f"model.{args.model}")
        except ImportError as exc:
            raise ValueError(f"Model '{args.model}' not found: {exc}") from exc
    except ValueError as exc:
        print(f"Error: {exc}")

    model_params = config["model_params"][args.model]

    # Path for the dataset
    base_path = config["dataset_path"]
    # List to store session paths
    subjects = []

    # Iterate over the 18 subjects
    for subject_number in range(1, 20):
        subject_path1 = f"subject{subject_number:02d}_session1/"
        subject_path2 = f"subject{subject_number:02d}_session2/"

        # Path of the first session
        session1_path = os.path.join(base_path, subject_path1)

        # Path of the second session
        session2_path = os.path.join(base_path, subject_path2)

        # Add session paths to the list
        subjects.append([session1_path, session2_path])
    # Build the model
    model = model_module.build_model(model_params)

    # View information about the model
    print(model.summary())

    model_dir = f"logs/{args.model}/na_all_fingers/"
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(model_dir)
        print("Model restored successfully.")
    else:
        print("Checkpoint directory does not exist or is not a directory.")

    BATCH_SIZE = config["BATCH_SIZE"]
    INPUT_SHAPE_0 = config["input_shape"][0]
    INPUT_SHAPE_1 = config["input_shape"][1]
    OUTPUT_SHAPE = config["OUTPUT_SHAPE"]
    EPOCHS = config["EPOCHS"]

    for i in range(len(subjects)):
        # Defines the complete model
        model = model_module.build_model(model_params)
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        print("Paciente: " + str(i + 1))
        path1 = [subjects[i][0]]
        path2 = [subjects[i][1]]
        path3 = [subjects[i][1]]
        train_dataset = (
            tf.data.Dataset.from_generator(
                lambda: create_synced_datasets_generator(
                    path1, emg_tr_files_lists, force_tr_files_lists
                ),
                output_signature=(
                    tf.TensorSpec(
                        shape=(INPUT_SHAPE_0, INPUT_SHAPE_1), dtype=tf.float32
                    ),
                    tf.TensorSpec(shape=(OUTPUT_SHAPE,), dtype=tf.float32),
                ),
            )
            .shuffle(buffer_size=BATCH_SIZE)
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_generator(
                lambda: create_synced_datasets_generator(
                    path2, emg_vl_files_lists, force_vl_files_lists
                ),
                output_signature=(
                    tf.TensorSpec(
                        shape=(INPUT_SHAPE_0, INPUT_SHAPE_1), dtype=tf.float32
                    ),
                    tf.TensorSpec(shape=(OUTPUT_SHAPE,), dtype=tf.float32),
                ),
            )
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        test_dataset = (
            tf.data.Dataset.from_generator(
                lambda: create_synced_datasets_generator(
                    path3, emg_te_files_lists, force_te_files_lists
                ),
                output_signature=(
                    tf.TensorSpec(
                        shape=(INPUT_SHAPE_0, INPUT_SHAPE_1), dtype=tf.float32
                    ),
                    tf.TensorSpec(shape=(OUTPUT_SHAPE,), dtype=tf.float32),
                ),
            )
            .batch(32)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        # Callbacks
        callbacks = get_callbacks(args.model, i)

        # Fit data to model
        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
        )
        mse, mae = model.evaluate(test_dataset)
        print("Test MSE:", mse)
        print("Test MAE:", mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model name")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path of the configuration file",
        default="configs/config.yml",
    )
    args = parser.parse_args()

    # Verify that the model name is provided as an argument
    if args.model:
        main(args)
    else:
        print("Please provide the model name using the -m or --model option.")
