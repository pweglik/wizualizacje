from enum import Enum


class DataType(Enum):
    MNIST = 1
    SVHN = 2
    CIFAR = 3


MODEL_PATH_PREFIX = "./generated/model/"
TSNE_PATH_PREFIX = "./generated/tsne/points_"
