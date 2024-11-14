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


def run_experiments(depths, generations=5, n_points=3000, split_ratio=1/3):
    """
    Run experiments for different max_depth values and report accuracies.
    """
    # Numpy array containing accuracy values over the following dimensions
    # The last dimension corresponds to the distinction between train accuracy and test accuracy
    accuracies = np.zeros((len(depths), generations, 2))
    
    for generation in range(generations):
        # Load dataset
        x, y = make_dataset(n_points)
        split = int(split_ratio * len(x))
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        for i_depth, depth in enumerate(depths):
            clf = train_decision_tree(x_train, y_train, depth)
            cm_test, accuracy_test = evaluate_model(clf, x_test, y_test)
            cm_train, accuracy_train = evaluate_model(clf, x_train, y_train)
            accuracies[i_depth, generation, 0] = accuracy_test
            accuracies[i_depth, generation, 1] = accuracy_train
            
            # Plot decision boundary
            # X_train_2d = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
            plot_boundary(f'results/dt_d{depth}', clf, x_train, y_train, mesh_step_size=0.1, title='Decision Tree with max_depth={}'.format(depth))

    # Calculate average and standard deviation of accuracies
    for i, depth in enumerate(depths):
        # Using numpy broadcasting to compute mean and variation
        test_mean_accuracy, train_mean_accuracy = np.mean(accuracies[i, :, :], axis=0)
        test_std_accuracy, train_std_accuracy = np.std(accuracies[i, :, :], axis=0)
        print(f"depth {depth}. Test [mean: {test_mean_accuracy:.4f}, std: {test_std_accuracy:.4f}]. Train [mean: {train_mean_accuracy:.4f}, std: {train_std_accuracy:.4f}]")

    plt.figure(figsize=(12, 6))

    labels = [str(depth) if depth is not None else 'Unspecified' for depth in depths]

    plt.subplot(1, 2, 1)
    plt.boxplot(accuracies[:, :, 0].T, tick_labels=labels)
    plt.title(f'Test set accuracy over {generations} generations')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.boxplot(accuracies[:, :, 1].T, tick_labels=labels)
    plt.title(f'Train set accuracy over {generations} generations')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'results/boxplots_accuracies_g{generations}.png')

if __name__ == "__main__":
    max_depth_values = [1, 2, 4, 8, None]
    run_experiments(max_depth_values)
