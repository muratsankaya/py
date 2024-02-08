import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from src import PolynomialRegression, KNearestNeighbor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def visualize_model(data, model, title):
    """
    This function produces a plot with the training and test datasets,
    as well as the predictions of the trained model. The plot is saved
    to `free_response/` as a png.

    Note: You should not need to change this function!

    Args:
        data (tuple of np.ndarray): four arrays containing, in order:
            training data, test data, training targets, test targets
        model: the model with a .predict() method
        title: the title for the figure

    Returns:
        train_mse (float): mean squared error on training data
        test_mse (float): mean squared error on test data
    """

    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    x_func = np.arange(-1.2, 1.2, 0.01).reshape(-1, 1)
    preds = model.predict(x_func)

    train_mse = mean_squared_error(
        y_train, model.predict(X_train))
    test_mse = mean_squared_error(
        y_test, model.predict(X_test))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)

    fig.suptitle(title)
    ax1.set_title(f"Train MSE: {train_mse:.2f}",
                  fontdict={"fontsize": "medium"})
    ax1.set_ylim(-20, 20)
    ax1.scatter(X_train, y_train)
    ax1.plot(x_func, preds, "orange", label="h(X)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()

    ax2.set_title(f"Test MSE: {test_mse:.2f}",
                  fontdict={"fontsize": "medium"})
    ax2.set_ylim(-20, 20)
    ax2.scatter(X_test, y_test)
    ax2.plot(x_func, preds, "orange", label="h(X)")
    ax2.set_xlabel('X')
    ax2.legend()
    plt.savefig(f"free_response/plots/{title}.png")
    plt.close()

    return train_mse, test_mse


def part_a_plot(model="regression"):
    """
    This uses matplotlib to create an example plot that you can modify
    for your answers to FRQ1 part a.
    """

    # Create a plot with four subplots
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)

    fig.suptitle(f"{model}")
    df = pd.read_csv(f"free_response/q1_{model}.csv")

    if model == "regression":
        x_key = "Degree"
    elif model == "knn":
        x_key = "K"
    else:
        raise ValueError(f"What is {model}?")

    # One subplot for each amount of data
    amounts = [8, 16, 64, 256]

    for idx, amount in enumerate(amounts):
        axes[idx].set_title(
            f"{amount} data points", fontdict={"fontsize": "small"}, pad=3)

        subdf = df[df["Amount"] == amount]

        x_axis = subdf[x_key]
        train_mse = subdf["Train MSE"]
        test_mse = subdf["Test MSE"]

        # Plot a solid red line for train error
        axes[idx].plot(np.array(x_axis), train_mse, 'r-', label="Train")
        # Plot a dashed blue line for test error
        axes[idx].plot(np.array(x_axis), test_mse, 'b--', label="Test")

        axes[idx].set_ylabel('MSE')
        axes[idx].legend()

    axes[idx].set_xlabel(x_key)
    plt.savefig(f"free_response/{model}.png")


def load_frq_data(amount):
    '''
    Loads the data provided for this free-response question,
    with `amount` examples.

    Note: You should not need to change this function!

    Args:
        amount (int): the number of examples to include
        in the dataset. Should be one of 8, 16, 64, or 256

    Returns
        data (tuple of np.ndarray): four arrays containing, in order:
            training data, test data, training targets, test targets
    '''
    df = pd.read_csv(f"data/frq.{amount}.csv")
    x1 = df[["x"]].to_numpy()
    y1 = df[["y"]].to_numpy()
    
    return train_test_split(
        x1, y1, train_size=0.8, random_state=0, shuffle=False)


def polynomial_regression_experiment():
    """
    Run 16 experiments with fitting PolynomialRegression models
        of different degrees on datasets of different sizes.

    You will want to use the `load_frq_data` and `visualize_model`
        functions, and may want to add some print statements to
        collect data on the overall trends.
    """
    degrees = [1, 2, 4, 16]
    amounts = [8, 16, 64, 256]

    with open("free_response/q1_regression.csv", "w") as outf:
        writer = csv.writer(outf)
        writer.writerow("Degree,Amount,Train MSE,Test MSE".split(","))
        for amount in amounts:
            for degree in degrees:
                title = f"{degree}-degree Regression with {amount} points"

                dataset = load_frq_data(amount)
                model = PolynomialRegression(degree)
                train_mse, test_mse = visualize_model(dataset, model, title)
                writer.writerow(f"{degree},{amount},{train_mse:.2f},{test_mse:.2f}".split(","))


def knn_regression_experiment():
    '''
    Run 16 experiments with fitting KNearestNeighbor models
        of different n_neighbors on datasets of different sizes.

    You will want to use the `load_frq_data` and `visualize_model`
        functions, and may want to add some print statements to
        collect data on the overall trends.

    Use Euclidean distance and Mean aggregation for all experiments.
    '''
    n_neighbors = [1, 2, 4, 8]
    amounts = [8, 16, 64, 256]
    with open("free_response/q1_knn.csv", "w") as outf:
        writer = csv.writer(outf)
        writer.writerow("K,Amount,Train MSE,Test MSE".split(","))
        for amount in amounts:
            for neighbor in n_neighbors:
                title = f"{neighbor}-NN with {amount} points"

                dataset = load_frq_data(amount)
                model = KNearestNeighbor(neighbor, distance_measure="euclidean", aggregator="mean")
                train_mse, test_mse = visualize_model(dataset, model, title)
                writer.writerow(f"{neighbor},{amount},{train_mse:.2f},{test_mse:.2f}".split(","))


if __name__ == "__main__":
    polynomial_regression_experiment()
    knn_regression_experiment()
    part_a_plot("regression")
    part_a_plot("knn")
    plt.close('all')
