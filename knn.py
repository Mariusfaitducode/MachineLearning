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

def run_experiments(n_neighbors_list, generations=5, n_points=1000):
    """
    Run experiments for different n_neighbors values and report accuracies.
    """
    accuracies = {n: [] for n in n_neighbors_list}
    
    for _ in range(generations):
        # Load dataset
        x, y = make_dataset(n_points)
        split = int(0.8 * len(x))
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        
        for n_neighbors in n_neighbors_list:
            clf = train_knn(x_train, y_train, n_neighbors)
            cm, accuracy = evaluate_model(clf, x_test, y_test)
            accuracies[n_neighbors].append(accuracy)
            
            # Plot decision boundary
            plot_boundary(f'knn_neighbors_{n_neighbors}', clf, x_train, y_train, mesh_step_size=0.1, title=f'KNN with n_neighbors={n_neighbors}')

    # Calculate average and standard deviation of accuracies
    for n_neighbors in n_neighbors_list:
        mean_accuracy = np.mean(accuracies[n_neighbors])
        std_accuracy = np.std(accuracies[n_neighbors])
        print(f"n_neighbors: {n_neighbors}, Average Accuracy: {mean_accuracy:.2f}, Std Dev: {std_accuracy:.2f}")


if __name__ == "__main__":
    n_neighbors_values = [1, 3, 5, 7, 9]
    run_experiments(n_neighbors_values)
