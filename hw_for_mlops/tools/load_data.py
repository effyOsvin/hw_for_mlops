import os
import sys

import numpy as np


def load_data(flatten=False, train=True):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source="http://yann.lecun.com/exdb/mnist/"):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return np.copy(data)

    if train:
        X = load_mnist_images("train-images-idx3-ubyte.gz")
        y = load_mnist_labels("train-labels-idx1-ubyte.gz")
    else:
        X = load_mnist_images("t10k-images-idx3-ubyte.gz")
        y = load_mnist_labels("t10k-labels-idx1-ubyte.gz")

    if flatten:
        X = X.reshape([X.shape[0], -1])

    return X, y
