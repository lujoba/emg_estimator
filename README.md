# SEMG force estimation study

A functions collection, created in my Ph.D. research, to estimate the muscle forces through s-EMG signals.

## Dependencies

- [tensorflow](https://www.tensorflow.org/): Deep Learning framework of choice, because of the facility of deployment on edge devices;
- [matplotlib](https://matplotlib.org/): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python;
- [wfdb](https://github.com/MIT-LCP/wfdb-python/): Python-native package for reading, writing, processing, and plotting physiologic signal and annotation data.
- [numpy](https://numpy.org/): The fundamental package for scientific computing with Python.
- [einops](https://github.com/arogozhnikov/einops): Einstein-Inspired Notation for operations, for easier manipulations;


## Notebooks

- [dilatation_conv_representation](notebooks\dilatation_conv_representation.ipynb): shows how dilatation convolution works;
- [online_estimator](notebooks\online_estimator.ipynb): update the linear combinator to an online estimator;
- [training_models](notebooks\training_models.ipynb): training the basic force estimator;
- [transferlearning_estimator](notebooks\transferlearning_estimators.ipynb): transferlearning script to finetuning the linear combinator layer

## Set Up

1. Check out the code;
2. Install requirements:
    ```
    python -m venv emg_estimation
    source emg_estimation/bin/activate
    pip install - r requirements.txt
    ```
3. Use as CLI, as the example below, or call the class in your project:
    ```
    python train.py -m model_name -c config_file_path
    ```
4. Dataset:
    ```
    wget -r -N -c -np https://physionet.org/files/hd-semg/1.0.0/
    ```
5. Cite me;
6. Have fun.

## To Do

- Deployement script
- GUI
