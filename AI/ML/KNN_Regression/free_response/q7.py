import csv
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from src import equalized_odds_difference, demographic_parity_difference
from src import matplotlib, plt
matplotlib.rc('font', family="serif", size=14)


def train_logistic_regression(X, y, group, where=None, name=''):
    """
    Train a logistic regression model and prints out fairness metrics.
        You should not need to modify this code to answer the questions.
        The model is defined as g(w_1 * x_1 + ... + w_k * x_k + b)
        where g is the logistic function.

    args:
        X: the features
        y: the label
        group: a column indicating which individuals belong to the protected subgroup
        where: if not None, only train on individuals where this array is True
        name: a helper string for printouts
    returns:
        logreg: the trained logistic regression model
    """

    if where is not None:
        X = X[where]
        y = y[where]
        group = group[where]

    logreg = LogisticRegression()
    logreg.fit(X, y)
    preds = logreg.predict(X)

    # Print out the model's coefficients and intercept.
    parameters = [f"{a:.3f}*{b}" for a, b in zip(logreg.coef_[0], X.columns)]
    intercept = logreg.intercept_[0]
    sign = "+" if intercept >= 0 else "-"
    print(f"{name} Model: g({' + '.join(parameters)} {sign} {np.abs(logreg.intercept_[0]):.3f})")

    # Print out the model's overall accuracy (across both groups)
    print(f"{name} Overall accuracy: {accuracy_score(y, preds)*100:.1f}%")

    confusion_matrices = {}
    for g in [0, 1]:
        where = (group == g)
        if np.any(where):
            confusion_matrices[g] = confusion_matrix(y[where], preds[where])

            # Print out metrics that only apply to group 'g'
            accuracy = accuracy_score(y[where], preds[where])
            precision = precision_score(y[where], preds[where])
            recall = recall_score(y[where], preds[where])
            print(f"{name} G={g} Accuracy: {accuracy*100:.1f}% Precision: {precision:.3f} Recall: {recall:.3f}")

    # Unless this model was built only on members of one group,
    # compute our fairness metrics across the two groups.
    if len(confusion_matrices) == 2:
        dpd = demographic_parity_difference(*confusion_matrices.values())
        eod = equalized_odds_difference(*confusion_matrices.values())
        print(f"{name} Demographic Parity Difference: {dpd:.3f}")
        print(f"{name} Equalized Odds Difference: {eod:.3f}")
        print()

    return logreg, confusion_matrices


def build_figure(data):
    '''
    Build a plot for the four visualizations
    '''
    nrow = 2
    ncol = 2
    fig, axes = plt.subplots(
        nrow, nrow, sharex=True, sharey=True, figsize=(8, 8))
    axes = axes.reshape(-1).tolist()

    # Create custom legend with shape and color
    def handle_plot(m, c):
        return axes[0].plot([], [], marker=m, color=c, ls="none")[0]

    cmap = plt.get_cmap("Set1")
    handle_args = [("s", cmap(0.0)), ("s", cmap(1.0)), ("x", "k"), ("o", "k")]
    handles = [handle_plot(*x) for x in handle_args]
    labels = ["No Loan", "Loan", "Group 0", "Group 1",]
    fig.legend(handles, labels, ncol=4, loc="upper center", framealpha=1.0)

    for i, axis in enumerate(axes):
        if i % 2 == 0:
            axis.set_ylabel("Credit")
        if i >= (nrow * ncol // 2):
            axis.set_xlabel("Income")

    return (fig, axes)


def visualize_data(data, ax, name=''):
    '''
    Visualize the data for this problem
    '''
    xmin, xmax = np.percentile(data["I"], [0, 100])
    ymin, ymax = np.percentile(data["C"], [0, 100])
    ax.set_xlim(xmin - 10, xmax + 10)
    ax.set_ylim(ymin - 10, ymax + 10)
    x_arr = np.array([xmin - 10, xmax + 10])

    group0 = data["G"] == 0
    group1 = data["G"] == 1

    ax.scatter(
       x=data["I"][group0], y=data["C"][group0], cmap="Set1", c=data["L"][group0], marker="x")
    ax.scatter(x=data["I"][group1], y=data["C"][group1], cmap="Set1", c=data["L"][group1], marker="o")


    if len(name) > 0:
        ax.set_title(name)


def visualize_logistic_regression(logreg, ax):
    '''
    Visualize the logistic regression model.
    For these problems, the positive side of the decision boundary is
        shaded grey, and the negative side is shaded red.
    '''

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_arr = np.array([xmin, xmax])

    b = logreg.intercept_[0]
    w1, w2 = logreg.coef_.T[:2]

    y_arr = (-w1 / w2) * x_arr - (b / w2)

    ax.plot(x_arr, y_arr, 'k', lw=1, ls='--')
    ax.fill_between(x_arr, y_arr, ymin - 20, color="tab:red", alpha=0.2)
    ax.fill_between(x_arr, y_arr, ymax + 20, color="tab:grey", alpha=0.2)


def part_a(data, ax):
    """
    Train a logistic regression to predict Loan using Income and Credit as features
    """
    name = "a."
    logreg, matrices = train_logistic_regression(data[["I", "C"]], data["L"], data["G"], name=name)
    visualize_data(data, ax, name)
    visualize_logistic_regression(logreg, ax)


def part_b(data, ax):
    """
    Train a logistic regression to predict Loan using Income, Credit, and Group as features
    """
    name = "b."
    logreg, matrices = train_logistic_regression(data[["I", "C", "G"]], data["L"], data["G"], name=name)
    visualize_data(data, ax, name)
    visualize_logistic_regression(logreg, ax)


def part_c(data, ax, group):
    """
    Train a logistic regression to predict Loan, using *only* on individuals where Group = {group}
    """
    name = f"c. G={group}"
    where = data["G"] == group
    logreg, matrices = train_logistic_regression(data[["I", "C"]], data["L"], data["G"], where=where, name=name)
    visualize_data(data, ax, name)
    visualize_logistic_regression(logreg, ax)
    return matrices


def main():
    data = pd.read_csv("data/frq.fairness.csv")

    fig, axes = build_figure(data)

    print()
    part_a(data, axes[0])
    part_b(data, axes[1])

    matrix_0 = part_c(data, axes[2], group=0)
    matrix_1 = part_c(data, axes[3], group=1)

    # Compute the fairness metrics between the two models
    # trained in part c.
    dpd = demographic_parity_difference(matrix_0[0], matrix_1[1])
    eod = equalized_odds_difference(matrix_0[0], matrix_1[1])
    print(f"c. Demographic Parity Difference: {dpd:.3f}")
    print(f"c. Equalized Odds Difference: {eod:.3f}")
    print()

    plt.savefig(f"free_response/fairness.png")
    plt.close("all")


if __name__ == "__main__":
  main()
