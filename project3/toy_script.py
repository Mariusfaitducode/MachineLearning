#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier


def load_data(data_path=None, max_size=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
    print(f"Using data path: {data_path}")
    FEATURES = range(2, 33)

    # Load full data first
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    
    # Get full data size from first sensor file
    full_data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_2.txt'))
    N_TIME_SERIES = min(max_size, len(full_data)) if max_size is not None else len(full_data)

    # Create the training and testing samples with proper size
    X_train = np.zeros((N_TIME_SERIES, (len(FEATURES) * 512)))
    X_test = np.zeros((N_TIME_SERIES, (len(FEATURES) * 512)))

    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data[:N_TIME_SERIES]
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data[:N_TIME_SERIES]
    
    # Load labels and subject IDs with proper size
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))[:N_TIME_SERIES]
    subject_ids_train = np.loadtxt(os.path.join(LS_path, 'subject_Id.txt'))[:N_TIME_SERIES]
    subject_ids_test = np.loadtxt(os.path.join(TS_path, 'subject_Id.txt'))[:N_TIME_SERIES]

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test, subject_ids_train, subject_ids_test


def write_submission(y, submission_path='example_submission.csv'):
    parent_dir = os.path.dirname(submission_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(submission_path):
        os.remove(submission_path)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))
    
    # Write submission file
    with open(submission_path, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')

        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    print(f'Submission saved to {submission_path}.')


if __name__ == '__main__':
    X_train, y_train, X_test, subject_ids_train, subject_ids_test = load_data()

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    y_test = clf.predict(X_test)

    write_submission(y_test)
