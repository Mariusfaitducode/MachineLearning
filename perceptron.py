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


class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5, learning_rate=.0001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.w = None
        self.b = None

    def fit(self, X, y):
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

        # Input validation
        X = np.asarray(X, dtype=np.float64)
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert y to {-1, 1}
        y = np.where(y == 0, -1, 1)

        # Training loop
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                update = self.learning_rate * yi * (1 if np.dot(xi, self.w) + self.b <= 0 else 0)
                self.w += update * xi
                self.b += update

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        X = np.asarray(X)
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, 0)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = np.asarray(X)
        scores = np.dot(X, self.w) + self.b
        
        # Clip scores to avoid overflow

        # We clip the scores to a range that won't cause overflow in the exponential function.
        # The maximum value that can be safely exponentiated for a 64-bit float is approximately e^709, 
        # so we clip the scores to the range [-709, 709].
        # By clipping the scores, we ensure that the exponential function won't overflow, 
        # and the probability calculation will be stable.

        scores = np.clip(scores, -709, 709)  # log(np.finfo(np.float64).max) ≈ 709
        
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack((1 - probs, probs))

if __name__ == "__main__":
    # Generate a dataset
    X, y = make_dataset(n_points=1000)

    # Split the dataset
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train the perceptron
    clf = PerceptronClassifier(n_iter=100, learning_rate=0.01)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot decision boundary
    plot_boundary("perceptron", clf, X, y, mesh_step_size=0.1, title="Perceptron Decision Boundary")
