import gzip
import os

import numpy as np


def load_mnist_images(filename):
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 1, 28, 28)


def load_mnist_labels(filename):
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return np.copy(data)


def load_data(train=True):
    if train:
        os.system("dvc pull " + "data/train-images.gz" + ".dvc")
        os.system("dvc pull " + "data/train-labels.gz" + ".dvc")
        X = load_mnist_images("data/train-images.gz")
        y = load_mnist_labels("data/train-labels.gz")
    else:
        os.system("dvc pull " + "data/test-images.gz" + ".dvc")
        os.system("dvc pull " + "data/test-labels.gz" + ".dvc")
        X = load_mnist_images("data/test-images.gz")
        y = load_mnist_labels("data/test-labels.gz")

    return X, y
