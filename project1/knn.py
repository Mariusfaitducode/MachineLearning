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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 2): KNN

def train_knn(X_train, y_train, n_neighbors):
    """
    Train a K-Nearest Neighbors Classifier with a specified number of neighbors.
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
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

def run_experiments(n_neighbors_list, generations=5, n_points=3000, split_ratio=1/3):
    """
    Run experiments for different n_neighbors values and report accuracies.
    """
    accuracies = np.zeros((len(n_neighbors_list), generations, 2))

    for generation in range(generations):
        # Load dataset
        x, y = make_dataset(n_points)
        split = int(split_ratio * len(x))
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        for i_n, n_neighbors in enumerate(n_neighbors_list):
            clf = train_knn(x_train, y_train, n_neighbors)
            cm_test, accuracy_test = evaluate_model(clf, x_test, y_test)
            cm_train, accuracy_train = evaluate_model(clf, x_train, y_train)
            accuracies[i_n, generation, 0] = accuracy_test
            accuracies[i_n, generation, 1] = accuracy_train
            
            # Plot decision boundary
            plot_boundary(f'knn_neighbors_{n_neighbors}', clf, x_train, y_train, mesh_step_size=0.1, title=f'KNN with n_neighbors={n_neighbors}')
            # plot_boundary(f'test_knn_neighbors_{n_neighbors}', clf, x_test, y_test, mesh_step_size=0.1, title=f'KNN with n_neighbors={n_neighbors}')

    # Calculate average and standard deviation of accuracies
    for i, n_neighbors in enumerate(n_neighbors_list):
        # Using numpy broadcasting to compute mean and variation
        test_mean_accuracy, train_mean_accuracy = np.mean(accuracies[i, :, :], axis=0)
        test_std_accuracy, train_std_accuracy = np.std(accuracies[i, :, :], axis=0)
        print(f"NN: {n_neighbors}. Test [mean: {test_mean_accuracy:.4f}, std: {test_std_accuracy:.4f}]. Train [mean: {train_mean_accuracy:.2f}, std: {train_std_accuracy:.2f}]")

    plt.figure(figsize=(12, 6))

    labels = [str(n_neighbors) if n_neighbors is not None else 'Unspecified' for n_neighbors in n_neighbors_list]

    plt.subplot(1, 2, 1)
    plt.boxplot(accuracies[:, :, 0].T, tick_labels=labels)
    plt.title(f'Test set accuracy over {generations} generations')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.boxplot(accuracies[:, :, 1].T, tick_labels=labels)
    plt.title(f'Train set accuracy over {generations} generations')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'results/nn_boxplots_accuracies_g{generations}.png')

if __name__ == "__main__":
    n_neighbors_values = [1, 5, 50, 100, 500]
    run_experiments(n_neighbors_values)
