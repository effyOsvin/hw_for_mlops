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


def load_data(X_path, y_path):
    if not os.path.exists(X_path):
        os.system("dvc pull " + X_path + ".dvc")
    if not os.path.exists(y_path):
        os.system("dvc pull " + y_path + ".dvc")
    X = load_mnist_images(X_path)
    y = load_mnist_labels(y_path)

    return X, y
