import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from network.constants import DataType
from network.constants import MODEL_PATH_PREFIX
import numpy as np
from sklearn.manifold import MDS
from sklearn.ensemble import ExtraTreesClassifier
import os


def create_multilayer_perceptron(dataType):
    if dataType == DataType.MNIST:
        input_shape = 784
    else:
        input_shape = 3072

    model = Sequential()
    model.add(Dense(1000, activation="relu", input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def create_cnn(dataType):
    if dataType == DataType.MNIST:
        input_shape = (28, 28, 1)
        neurons_count = 3136
    else:
        input_shape = (32, 32, 3)
        neurons_count = 4096

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(neurons_count, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def load_model_from_file(model_name, epochs):
    return load_model(
        os.path.join(
            MODEL_PATH_PREFIX, model_name, "model_" + model_name + f"_{epochs}"
        )
    )


def train_model(model, batch_size, epochs, model_name, X_train, Y_train):
    weights_path = (
        MODEL_PATH_PREFIX + model_name + "_" + str(epochs) + "_{epoch:2d}.hdf5"
    )
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, train_size=5 / 6
    )
    network_history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint],
    )

    model.save(MODEL_PATH_PREFIX + model_name + "_" + str(epochs))
    save_network_history_to_file(network_history, model_name, epochs)


def save_network_history_to_file(network_history, model_name, epochs):
    hist_df = pd.DataFrame(network_history.history)

    hist_json_file = (
        MODEL_PATH_PREFIX + model_name + "_" + str(epochs) + "_history.json"
    )
    with open(hist_json_file, mode="w") as file:
        hist_df.to_json(file)


def load_network_history_from_file(model_name, epochs):
    return pd.read_json(
        MODEL_PATH_PREFIX + model_name + "_" + str(epochs) + "_history.json"
    )


def predict_classes(model, X):
    return np.argmax(model.predict(X), axis=-1)


def load_weights_from_file(model, model_name, epochs, epoch):
    model.load_weights(
        MODEL_PATH_PREFIX
        + model_name
        + "/"
        + "model_"
        + model_name
        + "_"
        + str(epochs)
        + "_"
        + str(epoch)
        + ".hdf5"
    )


def create_neuron_projection(layer):
    coef = np.corrcoef(np.transpose(layer))
    for ix, iy in np.ndindex(coef.shape):
        coef[ix, iy] = 1 - abs(coef[ix, iy])
        if np.isnan(coef[ix, iy]):
            coef[ix, iy] = 0
    embedding = MDS(n_components=2, dissimilarity="precomputed")
    X_transformed = embedding.fit_transform(coef)
    return X_transformed
