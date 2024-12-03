import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_transformed_data

# Charger les données transformées
X_train, y_train, X_test = load_transformed_data()

# Séparer les données en ensembles d'entraînement, de validation et de test
# Premier split pour obtenir l'ensemble de test
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train
# )

print("Taille des ensembles:")
print(f"Entraînement: {X_train.shape[0]} échantillons")
# print(f"Validation: {X_val.shape[0]} échantillons") 
print(f"Test: {X_test.shape[0]} échantillons")


# Définir le pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Définir la grille de paramètres pour la recherche
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# Configurer la validation croisée et la recherche de grille
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=2)

# Entraîner le modèle
grid_search.fit(X_train, y_train)

# Évaluer le modèle sur l'ensemble de validation
# y_val_pred = grid_search.predict(X_val)
# print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
# print("Validation F1 Score:", f1_score(y_val, y_val_pred, average='weighted'))
# print("Classification Report:\n", classification_report(y_val, y_val_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# # Évaluer le modèle sur l'ensemble de test
# y_test_pred = grid_search.predict(X_test)
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
# print("Test F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))
# print("Classification Report:\n", classification_report(y_test, y_test_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Sauvegarder le meilleur modèle
joblib.dump(grid_search.best_estimator_, 'best_random_forest_model.pkl')
