import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
import scipy.io as sio
from network.constants import DataType


def load_data_mlp(dataType):
    def fix_0(digit):
        return 0 if digit == 10 else digit

    if dataType == DataType.MNIST:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784).astype("float32") / 255
        X_test = X_test.reshape(10000, 784).astype("float32") / 255
        Y_train = to_categorical(y_train, 10)
        Y_test = to_categorical(y_test, 10)
        return X_train, Y_train, X_test, Y_test

    elif dataType == DataType.SVHN:
        train_data = sio.loadmat("data/train_32x32.mat")
        test_data = sio.loadmat("data/test_32x32.mat")
        X_train = (
            np.swapaxes(train_data["X"], 0, 1).reshape(73257, 3072).astype("float32")
            / 255
        )
        y_train = np.array(list(map(fix_0, train_data["y"].reshape(-1))))
        X_test = (
            np.swapaxes(test_data["X"], 0, 1).reshape(26032, 3072).astype("float32")
            / 255
        )
        y_test = np.array(list(map(fix_0, test_data["y"].reshape(-1))))
        Y_train = to_categorical(y_train, 10)
        Y_test = to_categorical(y_test, 10)
        return X_train, Y_train, X_test, Y_test

    elif dataType == DataType.CIFAR:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(50000, 3072).astype("float32") / 255
        X_test = X_test.reshape(10000, 3072).astype("float32") / 255
        Y_train = to_categorical(y_train, 10)
        Y_test = to_categorical(y_test, 10)
        return X_train, Y_train, X_test, Y_test

    else:
        raise ValueError("Invalid dataType parameter")


def load_data_cnn(dataType):
    def fix_0(digit):
        return 0 if digit == 10 else digit

    if dataType == DataType.MNIST:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
        X_test = X_test.reshape((10000, 28, 28, 1)).astype("float32") / 255

    elif dataType == DataType.SVHN:
        train_data = sio.loadmat("data/train_32x32.mat")
        test_data = sio.loadmat("data/test_32x32.mat")

        X_train = np.transpose(train_data["X"], (3, 0, 1, 2)).astype("float32") / 255
        y_train = np.array(list(map(fix_0, train_data["y"].reshape(-1)))).astype(
            "float32"
        )
        X_test = np.transpose(test_data["X"], (3, 0, 1, 2)).astype("float32") / 255
        y_test = np.array(list(map(fix_0, test_data["y"].reshape(-1)))).astype(
            "float32"
        )

    elif dataType == DataType.CIFAR:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.astype("float32") / 255
        X_test = X_test.astype("float32") / 255

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test
