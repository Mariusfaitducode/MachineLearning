"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, accuracy_score

# (Question 3): Perceptron
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5, n_weights=2, learning_rate=.0001, activation=sigmoid):
        self.n_iter = n_iter
        self.n_weights = n_weights
        self.threshold = 0.5
        self.activation = activation
        self.learning_rate = learning_rate
        self.b = np.random.normal(loc=0, scale=10**-4)
        self.w = np.random.normal(loc=0, scale=10**-4, size=n_weights)
        #self.w = np.zeros(shape=n_weights)
        #self.b = 0

    def error(self, x, y_true, y_pred):
        return x * (y_pred - y_true)

    def epoch(self, X, y):
        random_indices = np.random.permutation(len(X))
        for i in random_indices:
            x, y_true = X[i], y[i]
        #for x, y_true in zip(X, y):
            y_pred = self.perceptron(x)
            self.b -= self.learning_rate * (y_pred - y_true)
            self.w -= self.learning_rate * self.error(x, y_true, y_pred)
        return self

    def fit(self, X, y) -> 'PerceptronClassifier':
        """Fit a perceptron model on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X = np.asarray(X, dtype=np.float64)

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary classification problems")

        # Training loop
        for _ in range(self.n_iter):
            self.epoch(X, y)

        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.where(self.perceptron(X) >= self.threshold, 1, 0)

    def perceptron(self, X):
        return self.activation(np.dot(X, self.w) + self.b)

    def predict_proba(self, X):
        X = np.asarray(X)
        probs = self.perceptron(X)
        return np.column_stack((probs, 1 - probs))


if __name__ == "__main__":
    # Generate a dataset
    split_ratio = 1/3
    generations = 5
    n_epochs = 5
    n_points = 3_000
    X, y = make_dataset(n_points=n_points, class_prop=.25)
    # Split the dataset
    split = int(split_ratio * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train the perceptron
    learning_rates = [10**-4, 5*10**-4, 10**-3, 10**-2, 10**-1]

    accuracies = np.zeros((len(learning_rates), generations, n_epochs, 2))
    for generation in range(generations):
        # Load dataset
        x, y = make_dataset(n_points)
        split = int(split_ratio * len(x))
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        for i_lr, lr in enumerate(learning_rates):
            clf = PerceptronClassifier(n_iter=n_epochs, n_weights=X.shape[1], learning_rate=lr)

            for epoch in range(n_epochs):
                clf = clf.epoch(x_train, y_train)
                y_test_pred = clf.predict(x_test)
                y_train_pred = clf.predict(x_train)
                accuracies[i_lr, generation, epoch,  0] = accuracy_score(y_true=y_test, y_pred=y_test_pred)
                accuracies[i_lr, generation, epoch, 1] = accuracy_score(y_true=y_train, y_pred=y_train_pred)

            # Plot decision boundary
            plot_boundary(
                f'perceptron_lr_{lr}',
                clf,
                x_train,
                y_train,
                mesh_step_size=0.1,
                title=f'Perceptron with learning_rate={lr}'
            )


    print(accuracies[:, :, -1, :], accuracies[:, :, -1, :].var(axis=0), accuracies[:, :, -1, :].mean(axis=0))
    labels = [str(lr) if lr is not None else 'Unspecified' for lr in learning_rates]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

    # LS accuracies plot
    for i, lr in enumerate(learning_rates):
        ax1.plot(range(n_epochs), accuracies[i, :, :, 0].mean(axis=0), marker='o', label=f"lr = {lr}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"LS mean accuracy ({generations} generations) over epoch")

    # TS accuracies plot
    for i, lr in enumerate(learning_rates):
        ax2.plot(range(n_epochs), accuracies[i, :, :, 1].mean(axis=0), marker='o', label=f"lr = {lr}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"TS mean accuracy ({generations} generations) over epoch")

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'results/perceptron_accuracies_g{generations}.png')