"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 1): Decision Trees

def train_decision_tree(X_train, y_train, max_depth):
    """
    Train a Decision Tree Classifier with a specified max_depth.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the model and return the confusion matrix and accuracy.
    """
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy


def run_experiments(depths, generations=5, n_points=1000):
    """
    Run experiments for different max_depth values and report accuracies.
    """
    accuracies = {depth: [] for depth in depths}
    
    for _ in range(generations):
        # Load dataset
        x, y = make_dataset(n_points)
        split = int(0.8 * len(x))
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        

        for depth in depths:
            clf = train_decision_tree(x_train, y_train, depth)
            cm, accuracy = evaluate_model(clf, x_test, y_test)
            accuracies[depth].append(accuracy)
            
            # Plot decision boundary
            # X_train_2d = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
            plot_boundary('decision_tree_depth_{}'.format(depth), clf, x_train, y_train, mesh_step_size=0.1, title='Decision Tree with max_depth={}'.format(depth))

    # Calculate average and standard deviation of accuracies
    for depth in depths:
        mean_accuracy = np.mean(accuracies[depth])
        std_accuracy = np.std(accuracies[depth])
        print(f"Max Depth: {depth}, Average Accuracy: {mean_accuracy:.2f}, Std Dev: {std_accuracy:.2f}")



if __name__ == "__main__":
    max_depth_values = [1, 2, 4, 8, None]
    run_experiments(max_depth_values)
