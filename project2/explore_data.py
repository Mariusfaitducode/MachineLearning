

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_wine_quality

# Charger les données
X, y = load_wine_quality()

# Créer un DataFrame pour une meilleure visualisation
# Les noms des features du Wine Quality dataset
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 
    'sulphates', 'alcohol'
]

df = pd.DataFrame(X, columns=feature_names)
df['quality'] = y

# Afficher les informations de base
print("=== Informations sur le dataset ===")
print(f"Nombre d'échantillons: {X.shape[0]}")
print(f"Nombre de features: {X.shape[1]}")
print("\nDistribution des scores de qualité:")
print(pd.Series(y).value_counts().sort_index())

# Statistiques descriptives
print("\n=== Statistiques descriptives ===")
print(df.describe())

# Visualisations
plt.figure(figsize=(15, 6))

# Distribution des scores de qualité
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='quality', bins=len(np.unique(y)))
plt.title('Distribution des scores de qualité')

# Boxplot des features
plt.subplot(1, 2, 2)
sns.boxplot(data=df[feature_names])
plt.xticks(rotation=45)
plt.title('Distribution des features (normalisées)')

plt.tight_layout()
plt.show()

# Matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.show()