from os.path import exists
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from datashader.bundling import hammer_bundle
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from keras.models import Model

from network.constants import TSNE_PATH_PREFIX
from network.data_loader import *
from network.network import *


def plot_history(network_history):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(network_history["loss"])
    plt.plot(network_history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(network_history["accuracy"])
    plt.plot(network_history["val_accuracy"])
    plt.legend(["Training", "Validation"], loc="lower right")
    plt.show()


def get_activations_mlp(model, data, size):
    layer_indices = [0, 2, 4, 6]
    layer_output = Model(inputs=model.inputs,
        outputs=[model.layers[i].output for i in layer_indices]
    )
    return layer_output([data[:size, :]])


def get_activations_cnn(model, data, size=None):
    layer_indices = [9, 10]
    layer_output = Model(inputs=model.inputs,
        outputs=[model.layers[i].output for i in layer_indices]
    )
    return layer_output([data[:size, :]])


def show_tsne(
    model_name: str,
    epochs: int,
    X: np.ndarray,
    Y: np.ndarray,
    Y_predicted: Optional[np.ndarray] = None,
    init: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    data = scaler.fit_transform(X)
    targets = np.argmax(Y, axis=1)

    file_path = f"{TSNE_PATH_PREFIX}{model_name}_{epochs}.npy"

    if init is not None:
        tsne = TSNE(n_components=2, perplexity=30, init=init, random_state=0)
    else:
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    points_transformed = tsne.fit_transform(data).T
    points_transformed = np.swapaxes(points_transformed, 0, 1)
    np.save(file_path, points_transformed)

    show_scatterplot(points_transformed, targets, Y_predicted)

    return points_transformed, targets


def show_scatterplot(points_transformed, targets, Y_predicted=None):
    palette = sns.color_palette("bright", 10)
    fig, ax = plt.subplots(figsize=(10, 10))
    if Y_predicted is None:
        sns.scatterplot(
            x=points_transformed[:, 0],
            y=points_transformed[:, 1],
            hue=targets,
            legend="full",
            palette=palette,
            ax=ax,
        )
    else:
        Y_diff = targets - Y_predicted
        styles = np.where(Y_diff == 0, "Matched", "Mismatched")
        sns.scatterplot(
            x=points_transformed[:, 0],
            y=points_transformed[:, 1],
            hue=targets,
            style=styles,
            legend="full",
            palette=palette,
            ax=ax,
        )
    plt.show()


def plot_new_neuron_projection(x, hue=None):
    palette = sns.color_palette("magma_r", as_cmap=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=hue, ax=ax, palette=palette)


def compare_projections(datatype, model_name, n_layer, label, size=2000):
    def get_model_and_activations():
        if "mlp" in model_name:
            load_data_func = load_data_mlp
            create_model_func = create_multilayer_perceptron
            get_activations_func = get_activations_mlp
        elif "cnn" in model_name:
            load_data_func = load_data_cnn
            create_model_func = create_cnn
            get_activations_func = get_activations_cnn
        else:
            raise ValueError(f"Unrecognized model name: {model_name}")

        X_train, Y_train, X_test, Y_test = load_data_func(datatype)
        model_bt = create_model_func(datatype)
        layer_bt = get_activations_func(model_bt, X_test, size)[n_layer - 1]
        x_bt = create_neuron_projection(layer_bt[:size])

        model_at = create_model_func(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_func(model_at, X_test, size)[n_layer - 1]
        x_at = create_neuron_projection(layer_at[:size])

        return model_bt, model_at, layer_bt, layer_at, x_bt, x_at, Y_test

    (
        model_bt,
        model_at,
        layer_bt,
        layer_at,
        x_bt,
        x_at,
        Y_test,
    ) = get_model_and_activations()

    label_test = np.argmax(Y_test[:size], axis=1) == label

    # TODO
    # Generate hues as mentioned in the article -> section 6.1
    # Use parameters layer_bt, layer_at and label_test
    # Finally, use plot_new_neuron_projection function


    classificator = ExtraTreesClassifier()
    classificator.fit(layer_bt[:size], label_test)
    hue_bt = classificator.feature_importances_

    classificator = ExtraTreesClassifier()
    classificator.fit(layer_at[:size], label_test)
    hue_at = classificator.feature_importances_

    plot_new_neuron_projection(x_bt, hue=hue_bt)
    plot_new_neuron_projection(x_at, hue=hue_at)


def plot_discriminative_map(activations, Y_test, size):
    palette = sns.color_palette("bright", 10)
    lst = []
    for digit in range(10):
        digit_test = np.argmax(Y_test[:size], axis=1) == digit
        etc = ExtraTreesClassifier()
        etc.fit(activations, digit_test)
        lst.append(etc.feature_importances_)
    arr = np.dstack(lst)
    labels = np.argmax(arr, axis=2).flatten()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    projection = create_neuron_projection(activations)

    def norm(x):
        return (x - np.min(x, axis=1)) / (np.max(x, axis=1) - np.min(x, axis=1))

    for n in range(10):
        projection_part = projection[labels == n]
        saturation = norm(arr[:, labels == n, n])
        sns.scatterplot(
            x=projection_part[:, 0],
            y=projection_part[:, 1],
            alpha=saturation,
            ax=ax,
            label=n,
            palette=palette,
        )


def compare_discriminative_map(datatype, model_name, n_layer, size):
    if "mlp" in model_name:
        X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
        model_bt = create_multilayer_perceptron(datatype)
        layer_bt = get_activations_mlp(model_bt, X_test, size)[n_layer - 1]

        model_at = create_multilayer_perceptron(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_mlp(model_at, X_test, size)[n_layer - 1]
    elif "cnn" in model_name:
        X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
        model_bt = create_cnn(datatype)
        layer_bt = get_activations_cnn(model_bt, X_test, size)[n_layer - 1]

        model_at = create_cnn(datatype)
        load_weights_from_file(model_at, model_name, 100, 100)
        layer_at = get_activations_cnn(model_at, X_test, size)[n_layer - 1]
    plot_discriminative_map(layer_bt, Y_test, size)
    plot_discriminative_map(layer_at, Y_test, size)


def show_seq_projections(datatype, model_name, n_layer, epoch, size):
    init = None
    if "mlp" in model_name:
        X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
        model_at = create_multilayer_perceptron(datatype)
        load_weights_from_file(model_at, model_name, 100, epoch)
        layer_at = get_activations_mlp(model_at, X_test, size)[n_layer]
    elif "cnn" in model_name:
        X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
        model_at = create_cnn(datatype)
        load_weights_from_file(model_at, model_name, 100, epoch)
        layer_at = get_activations_cnn(model_at, X_test, size)[n_layer - 1]

    file_path = TSNE_PATH_PREFIX + model_name + "_" + str(epoch) + ".npy"
    init = None
    if os.path.exists(file_path):
        init = np.load(file_path)

    Y_predicted = predict_classes(model_at, X_test)

    transformed_points, targets = show_tsne(
        model_name + f"_l{n_layer}",
        epoch,
        layer_at,
        Y_test[:size],
        Y_predicted[:size],
        init,
    )

    return transformed_points, targets


def inter_layer_evolution(points_lst, targets):
    # group points by target
    points_by_target = [[] for _ in range(10)]
    for n in range(10):
        points_by_layer = [points[targets == n] for points in points_lst]
        points_by_target[n] = points_by_layer

    fig, ax = plt.subplots(figsize=(15, 10))

    for label, epoch_points in enumerate(points_by_target):
        activation_lst = [
            [points[i, :] for points in epoch_points]
            for i in range(epoch_points[0].shape[0])
        ]
        dfs_lst = [
            pd.DataFrame(
                {f"activation_{j}_layer_{i}": value for i, value in enumerate(lst, 1)}
            )
            for j, lst in enumerate(activation_lst, 1)
        ]
        graph = nx.Graph()
        for df in dfs_lst:
            df_dense = df.corr("pearson")
            for edge, _ in df_dense.unstack().items():
                if int(edge[0].split("_")[-1]) + 1 == int(edge[1].split("_")[-1]):
                    graph.add_edge(*edge)
        c_dct = {
            f"activation_{j}_layer_{i}": value
            for i, layer_value in enumerate(epoch_points, 1)
            for j, value in enumerate(layer_value, 1)
        }
        nodes = (
            pd.DataFrame(c_dct)
            .T.reset_index()
            .rename(columns={"index": "name", 0: "x", 1: "y"})
        )
        sources = [nodes[nodes["name"] == source].index[0] for source, _ in graph.edges]
        targets = [nodes[nodes["name"] == target].index[0] for _, target in graph.edges]
        edges = pd.DataFrame({"source": sources, "target": targets})
        hb = hammer_bundle(nodes, edges)
        hb.plot(x="x", y="y", figsize=(10, 10), ax=ax, alpha=0.7, label=label)


def get_all_activations(datatype, model_name, n_layer, size):
    activations = []
    for epoch in [0, 20, 40, 60, 80, 100]:
        if "mlp" in model_name:
            X_train, Y_train, X_test, Y_test = load_data_mlp(datatype)
            model_at = create_multilayer_perceptron(datatype)
            if epoch:
                load_weights_from_file(model_at, model_name, 100, epoch)
            layer_at = get_activations_mlp(model_at, X_test, size)[n_layer - 1]
        elif "cnn" in model_name:
            X_train, Y_train, X_test, Y_test = load_data_cnn(datatype)
            model_at = create_cnn(datatype)
            if epoch:
                load_weights_from_file(model_at, model_name, 100, epoch)
            layer_at = get_activations_cnn(model_at, X_test, size)[n_layer - 1]
        activations.append(layer_at)
    return activations, Y_test[:size]


def show_whole_tsne(X, Y):
    data = StandardScaler().fit_transform(X)
    targets = np.argmax(Y, axis=1)
    points_transformed = (
        TSNE(n_components=2, perplexity=30, random_state=np.random.RandomState(0))
        .fit_transform(data)
        .T
    )
    points_transformed = np.swapaxes(points_transformed, 0, 1)
    show_scatterplot(points_transformed, targets)
    return points_transformed, targets


def process_activations(activations, Y_test, size):
    arr_activations = np.concatenate(activations, axis=0)
    arr_targets = np.concatenate([Y_test for _ in range(6)], axis=0)

    points_transformed, targets = show_whole_tsne(arr_activations, arr_targets)

    points_lst = list(points_transformed.reshape(6, size, 2))
    return points_lst, targets[:size]


def inter_epoch_evolution(points_lst, targets):
    points_by_digit = [[] for _ in range(10)]
    for n in range(10):
        for points in points_lst:
            ind = targets == n
            points_by_digit[n].append(points[ind])
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for label, epoch_points in enumerate(points_by_digit):
        activation_lst = []
        for n in range(epoch_points[0].shape[0]):
            epoch_lst = [
                epoch_points[count][n, :] for count in range(len(epoch_points))
            ]
            activation_lst.append(epoch_lst)
        dfs_lst = []
        for activation_count, lst in enumerate(activation_lst, 1):
            dct = {
                f"activation_{activation_count}_epoch_{count * 20}": value
                for count, value in enumerate(lst)
            }
            df = pd.DataFrame(dct)
            dfs_lst.append(df)
        graph = nx.Graph()
        for df in dfs_lst:
            df_dense = df.corr("pearson")
            for edge, _ in df_dense.unstack().items():
                if int(edge[0].split("_")[-1]) + 20 == int(edge[1].split("_")[-1]):
                    graph.add_edge(*edge)
        c_dct = {
            f"activation_{activation_count}_epoch_{epoch_count * 20}": activation_value
            for epoch_count, epoch_value in enumerate(epoch_points)
            for activation_count, activation_value in enumerate(epoch_value, 1)
        }
        nodes = (
            pd.DataFrame(c_dct)
            .T.reset_index()
            .rename(columns={"index": "name", 0: "x", 1: "y"})
        )
        sources = [
            nodes[nodes["name"] == source].index[0] for source, _ in list(graph.edges)
        ]
        targets = [
            nodes[nodes["name"] == target].index[0] for _, target in list(graph.edges)
        ]
        edges = pd.DataFrame({"source": sources, "target": targets})
        hb = hammer_bundle(nodes, edges)
        hb.plot(x="x", y="y", figsize=(10, 10), ax=ax, alpha=0.7, label=label)


def get_tsne(model_name, epochs, X, Y, init=None):
    data = StandardScaler().fit_transform(X)
    targets = np.argmax(Y, axis=1)

    file_path = TSNE_PATH_PREFIX + model_name + "_" + str(epochs) + ".npy"
    if exists(file_path):
        points_transformed = np.load(file_path)
    else:
        if init is not None:
            points_transformed = (
                TSNE(
                    n_components=2,
                    perplexity=30,
                    init=init,
                    random_state=np.random.RandomState(0),
                )
                .fit_transform(data)
                .T
            )
            np.save(file_path, points_transformed)
        else:
            points_transformed = (
                TSNE(
                    n_components=2, perplexity=30, random_state=np.random.RandomState(0)
                )
                .fit_transform(data)
                .T
            )
            np.save(file_path, points_transformed)
    points_transformed = np.swapaxes(points_transformed, 0, 1)

    return points_transformed, targets


def show_tsne_epoch_trace(model_name, datatype):
    epoch_points_transformed = []

    if "mlp" in model_name:
        _, _, X_test, Y_test = load_data_mlp(datatype)
        create_model = lambda: create_multilayer_perceptron(datatype)
        get_activations = lambda m, d: get_activations_mlp(m, d, 2000)[-1]
    elif "cnn" in model_name:
        _, _, X_test, Y_test = load_data_cnn(datatype)
        create_model = lambda: create_cnn(datatype)
        get_activations = lambda m, d: get_activations_cnn(m, d)[1]

    targets = np.argmax(Y_test[:2000], axis=1)

    # for untrained model
    model = create_model()
    l = get_activations(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    initial_points, _ = show_tsne(
        model_name + "_last_layer", 0, l, Y_test[:2000], Y_predicted[:2000]
    )

    epoch_points_transformed.append(initial_points)

    for epochs in [20, 40, 60, 80, 100]:
        model = create_model()
        load_weights_from_file(model, model_name, 100, epochs)
        l = get_activations(model, X_test)
        Y_predicted = predict_classes(model, X_test)
        points_transformed, _ = show_tsne(
            model_name + "_last_layer",
            epochs,
            l,
            Y_test[:2000],
            Y_predicted[:2000],
            initial_points,
        )

        epoch_points_transformed.append(points_transformed)

    show_trace(epoch_points_transformed, targets)


def show_trace(points, targets):
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(points[0])):
        for p in range(len(points) - 1):
            color = cmap(targets[i])
            alpha = 2 ** (1 + p - len(points))
            color = (*color[:3], alpha)
            xs = [points[p][i][0], points[p + 1][i][0]]
            ys = [points[p][i][1], points[p + 1][i][1]]
            ax.plot(xs, ys, c=color)

    plt.show()


def show_tsne_layer_trace(model_name, datatype):
    _, _, X_test, Y_test = load_data_mlp(datatype)

    model = create_multilayer_perceptron(datatype)
    load_weights_from_file(model, model_name, 100, 100)
    activations = get_activations_mlp(model, X_test, 2000)

    Y_predicted = predict_classes(model, X_test)
    targets = np.argmax(Y_test[:2000], axis=1)

    initial_points, _ = show_tsne(
        model_name + "_l1", 100, activations[0], Y_test[:2000], Y_predicted[:2000]
    )
    epoch_points_transformed = [initial_points]

    for layer_index, layer in enumerate(activations[1:], start=2):
        transformed_points, _ = show_tsne(
            model_name + f"_l{layer_index}",
            100,
            layer,
            Y_test[:2000],
            Y_predicted[:2000],
            initial_points,
        )
        epoch_points_transformed.append(transformed_points)

    show_trace(epoch_points_transformed, targets)
