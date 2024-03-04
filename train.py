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
from models.deep_lstm import build_model


def training(args):
    """
    Main function to train the various model arquitectures with the Hyser dataset.

    Args:
        args: argparse arguments containing configuration options and the model name.

    Raises:
        ValueError: If the model name is not found or arguments are invalid.

    Usage:
        python train.py -m model_name -c path_to_confi_file
    """
    # TODO use number of fingers as argument

    # Importing model module
    try:
        model_module = importlib.import_module(f"models.{args.model}")
    except ImportError as exc:
        raise ValueError(f"Model '{args.model}' not found: {exc}") from exc
    # Loads model configuration parameters from a YAML file.
    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"Error: {exc}")

    model_params = config["model_params"][args.model]
    model_params_dict = {}
    for param_dict in model_params:
        model_params_dict.update(param_dict)

    # Path for the dataset
    base_path = config["dataset_path"]
    # List to store session paths
    subjects = []

    # Iterate over the 20 subjects
    for subject_number in range(1, 21):
        subject_path1 = f"subject{subject_number:02d}_session1/"
        subject_path2 = f"subject{subject_number:02d}_session2/"

        # Path of the first session
        session1_path = os.path.join(base_path, subject_path1)

        # Path of the second session
        session2_path = os.path.join(base_path, subject_path2)

        # Add session paths to the list
        subjects.append([session1_path, session2_path])
    # Build the model base architecture
    backbone_model = model_module.build_model(**model_params_dict)
    estimator_model = build_model(embed_dim=256)
    
    #Parameters
    BATCH_SIZE = config["batch_size"]
    INPUT_SHAPE_0 = config["input_shape"][0]
    INPUT_SHAPE_1 = config["input_shape"][1]
    OUTPUT_SHAPE = config["output_shape"]
    EPOCHS = config["epoch"]
    DA = config["data_augment"]

    # Build the final model
    inputs = tf.keras.layers.Input(shape=(INPUT_SHAPE_0, INPUT_SHAPE_1))
    latent = backbone_model(inputs)
    outputs = estimator_model(latent)
    model = tf.keras.Model(inputs, outputs)

    # View information about the model
    print(model.summary())

    # Base folder for the model logs
    if DA:
        base_dir = f"logs/models/{args.model}/wa_all_fingers/"
    else:
        base_dir = f"logs/models/{args.model}/na_all_fingers/"

    # Check if the base directory exists and is a directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if os.path.exists(base_dir) and os.path.isdir(base_dir):
        # List all directories inside the base directory
        sub_dirs = os.listdir(base_dir)
        
        # List to store the found subject numbers
        found_subjects = []
        
        # Loop over the found directories
        for sub_dir in sub_dirs:
            # Check if the directory is a valid directory (not a file)
            if os.path.isdir(os.path.join(base_dir, sub_dir)):
                try:
                    # Try to convert the directory name to a subject number
                    subject_number = int(sub_dir.split('_')[1])
                    found_subjects.append(subject_number)
                except ValueError:
                    # If unable to convert to number, skip this directory
                    pass
        
        # Check if all 20 subjects are present
        if len(found_subjects) == 20:
            print("All 20 subjects are present.")
        elif found_subjects:
            print("Missing subjects. Last found subject number is:", max(found_subjects) + 1)
            j = max(found_subjects)
        else:
            print("No subjects found.")
            j = 0
        # Loop through to start with the next missing subject number
        for i in range(j, 20):
            if i < len(subjects):
                print("Starting processing for the next subject number:", i + 1)
                # Defines the complete model
                model = tf.keras.Model(inputs, outputs)
                if DA:
                    pi = tf.constant(3.141592653589793)
                    data_augmentation = tf.keras.Sequential([
                        tf.keras.layers.GaussianNoise(stddev=0.02),
                        AmplitudeScalingLayer(scale_range=(0.8, 1.2)),
                        TimeShiftLayer(max_shift=0.2),
                        PhaseRotationLayer(rotation_range=(-pi/9, pi/9)),
                        tf.keras.layers.Dropout(0.2),
                    ])
                    
                    model = tf.keras.Sequential([
                        data_augmentation,
                        model,
                    ])
                # Compile the model
                optimizer = keras.optimizers.Adam(learning_rate=1e-3)
                model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
                # Creat a path for the dataset
                path1 = [subjects[i][0]]
                path2 = [subjects[i][1]]
                path3 = [subjects[i][1]]

                # Generate the Training Dataset
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

                # Generate the Validation Dataset
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

                # Generate the Test Dataset
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

                # Load callbacks
                callbacks = get_callbacks(args.model, i + 1)

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
            else:
                print(f"Index {i + 1} is out of range. Skipping processing for this subject.")

    else:
        print("The checkpoint directory does not exist or is not a valid directory.")


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
        training(args)
    else:
        print("Please provide the model name using the -m or --model option.")
