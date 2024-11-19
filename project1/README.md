# Projet de Classification en Machine Learning

## 1. Arbres de Décision (Decision Trees)

### 1.1 Principe du modèle

Un arbre de décision est un modèle de machine learning qui prend des décisions en fonction d'une série de questions, similaire à un jeu de devinettes. Imaginez que vous essayez de classer des fruits :

1. Le fruit est-il rond ?
   - Si oui : Est-il orange ?
     - Si oui : C'est probablement une orange
     - Si non : Est-il rouge ?
       - Si oui : C'est probablement une pomme
       - Si non : ...
   - Si non : Est-il long ?
     - Si oui : C'est probablement une banane
     - Si non : ...

L'arbre continue ainsi, posant des questions et se ramifiant jusqu'à ce qu'il atteigne une conclusion (une classification).

### 1.2 Détails du modèle

- **Nœuds** : Chaque point de décision dans l'arbre.
- **Feuilles** : Les nœuds finaux qui donnent une classification.
- **Profondeur** : La longueur du chemin le plus long de la racine à une feuille.
- **max_depth** : Un paramètre qui limite la profondeur de l'arbre pour éviter le surapprentissage.

### 1.3 Structure et détails du code (dt.py)

Le fichier `dt.py` implémente et teste un classifieur basé sur les arbres de décision. Voici une explication détaillée :

1. **Importations** :
   - `numpy` : Pour les calculs numériques.
   - `matplotlib.pyplot` : Pour créer des graphiques.
   - `DecisionTreeClassifier` : L'implémentation de l'arbre de décision de scikit-learn.
   - Fonctions personnalisées pour générer des données et tracer des frontières de décision.

2. **Fonctions principales** :

   a. `train_decision_tree(X_train, y_train, max_depth)` :
   - Crée et entraîne un arbre de décision avec une profondeur maximale spécifiée.
   - Utilise les données d'entraînement (X_train, y_train) pour apprendre les règles de décision.

   b. `evaluate_model(clf, X_test, y_test)` :
   - Évalue les performances du modèle sur un ensemble de test.
   - Calcule la matrice de confusion et le score de précision.

   c. `run_experiments(depths, generations, n_points, split_ratio)` :
   - Effectue des expériences avec différentes profondeurs d'arbre.
   - Génère plusieurs jeux de données pour obtenir des résultats statistiquement significatifs.
   - Calcule et affiche les précisions moyennes et les écarts-types pour chaque profondeur.
   - Crée des graphiques pour visualiser les résultats.

3. **Processus principal** :
   - Génère un jeu de données.
   - Divise les données en ensembles d'entraînement et de test.
   - Entraîne le modèle avec différentes profondeurs.
   - Évalue le modèle sur les ensembles d'entraînement et de test.
   - Trace la frontière de décision pour chaque profondeur.
   - Crée des boîtes à moustaches pour visualiser la distribution des précisions.

4. **Visualisations** :
   - Frontières de décision : Montrent comment le modèle sépare les classes dans l'espace des caractéristiques.
   - Boîtes à moustaches : Illustrent la distribution des précisions pour différentes profondeurs d'arbre.

### 1.4 Interprétation des résultats

- Les arbres plus profonds (max_depth plus élevé) peuvent capturer des relations plus complexes dans les données.
- Cependant, des arbres trop profonds risquent de surapprendre, c'est-à-dire de trop bien s'adapter aux données d'entraînement au détriment de la généralisation.
- L'objectif est de trouver une profondeur qui offre un bon équilibre entre la précision sur l'ensemble d'entraînement et celle sur l'ensemble de test.

En exécutant ce code, vous pouvez observer comment la performance du modèle change avec différentes profondeurs d'arbre, vous aidant ainsi à choisir la meilleure configuration pour votre problème de classification.



## 2. K Plus Proches Voisins (K-Nearest Neighbors, KNN)

### 2.1 Principe du modèle

Le modèle des k plus proches voisins (k-NN) est une méthode de classification simple mais efficace. Son principe est basé sur l'idée que des éléments similaires sont généralement proches les uns des autres dans l'espace des caractéristiques.

Imaginez que vous voulez déterminer si un nouveau fruit est une pomme ou une orange :

1. Vous avez une collection de fruits déjà classés (pommes et oranges).
2. Pour chaque fruit, vous connaissez certaines caractéristiques (par exemple, la taille et la couleur).
3. Quand un nouveau fruit arrive, vous regardez les k fruits les plus proches en termes de caractéristiques.
4. Vous classez le nouveau fruit selon la majorité de ces k voisins.

### 2.2 Détails du modèle

- **k** : Le nombre de voisins à considérer pour la classification.
- **Distance** : Généralement la distance euclidienne entre les points dans l'espace des caractéristiques.
- **Règle de décision** : La classe majoritaire parmi les k voisins les plus proches.

### 2.3 Structure et détails du code (knn.py)

Le fichier `knn.py` implémente et teste un classifieur basé sur les k plus proches voisins. Voici une explication détaillée :

1. **Importations** :
   - `numpy` : Pour les calculs numériques.
   - `matplotlib.pyplot` : Pour créer des graphiques.
   - `KNeighborsClassifier` : L'implémentation k-NN de scikit-learn.
   - Fonctions personnalisées pour générer des données et tracer des frontières de décision.

2. **Fonctions principales** :

   a. `train_knn(X_train, y_train, n_neighbors)` :
   - Crée et entraîne un classifieur k-NN avec un nombre spécifié de voisins.
   - Utilise les données d'entraînement (X_train, y_train) pour apprendre le modèle.

   b. `evaluate_model(clf, X_test, y_test)` :
   - Évalue les performances du modèle sur un ensemble de test.
   - Calcule la matrice de confusion et le score de précision.

   c. `run_experiments(n_neighbors_list, generations, n_points, split_ratio)` :
   - Effectue des expériences avec différentes valeurs de k (nombre de voisins).
   - Génère plusieurs jeux de données pour obtenir des résultats statistiquement significatifs.
   - Calcule et affiche les précisions moyennes et les écarts-types pour chaque valeur de k.
   - Crée des graphiques pour visualiser les résultats.

3. **Processus principal** :
   - Génère un jeu de données.
   - Divise les données en ensembles d'entraînement et de test.
   - Entraîne le modèle avec différentes valeurs de k.
   - Évalue le modèle sur les ensembles d'entraînement et de test.
   - Trace la frontière de décision pour chaque valeur de k.
   - Crée des boîtes à moustaches pour visualiser la distribution des précisions.

4. **Visualisations** :
   - Frontières de décision : Montrent comment le modèle sépare les classes dans l'espace des caractéristiques.
   - Boîtes à moustaches : Illustrent la distribution des précisions pour différentes valeurs de k.

### 2.4 Interprétation des résultats

- Un k plus petit (par exemple, k=1) peut capturer des détails fins mais risque de surapprendre au bruit dans les données.
- Un k plus grand lisse la frontière de décision et peut mieux généraliser, mais risque de perdre des détails importants.
- L'objectif est de trouver une valeur de k qui offre un bon équilibre entre la précision sur l'ensemble d'entraînement et celle sur l'ensemble de test.

En exécutant ce code, vous pouvez observer comment la performance du modèle change avec différentes valeurs de k, vous aidant ainsi à choisir la meilleure configuration pour votre problème de classification.



## 3. Perceptron

### 3.1 Principe du modèle

Le Perceptron est un modèle fondamental en apprentissage automatique, servant de base aux réseaux de neurones plus complexes. Son objectif est de trouver une frontière linéaire qui sépare deux classes dans un espace multidimensionnel.

### 3.2 Fonctionnement détaillé du modèle

1. **Initialisation**:
   Le Perceptron commence avec des poids aléatoires ou nuls pour chaque caractéristique d'entrée, ainsi qu'un biais. Ces poids représentent l'importance de chaque caractéristique dans la décision de classification.

2. **Processus d'apprentissage**:
   - Pour chaque exemple d'entraînement, le Perceptron calcule une somme pondérée des caractéristiques d'entrée.
   - Si cette somme est supérieure à un seuil (généralement 0), le Perceptron prédit la classe positive, sinon la classe négative.
   - Si la prédiction est incorrecte, les poids sont ajustés dans la direction qui aurait donné la bonne réponse.
   - Ce processus est répété pour un nombre fixe d'itérations ou jusqu'à ce que tous les exemples soient correctement classés.

3. **Règle de mise à jour**:
   La règle de mise à jour du Perceptron est conçue pour minimiser l'erreur de classification. Elle ajuste les poids proportionnellement à l'erreur et à la valeur d'entrée. Cette approche est motivée par l'idée de renforcer les connexions qui conduisent à des prédictions correctes et d'affaiblir celles qui mènent à des erreurs.

4. **Conversion des étiquettes**:
   Les étiquettes sont converties en {-1, 1} au lieu de {0, 1} pour simplifier la règle de mise à jour. Avec cette conversion, la direction de l'ajustement est automatiquement déterminée par le signe de l'étiquette.

5. **Prédiction**:
   Pour faire une prédiction, le Perceptron calcule simplement la somme pondérée des entrées et applique une fonction de seuil. Cette approche est computationnellement efficace et directement liée à la nature linéaire du modèle.

6. **Estimation des probabilités**:
   Bien que le Perceptron soit fondamentalement un classificateur binaire, on peut obtenir des estimations de probabilité en appliquant une fonction sigmoïde à la sortie linéaire. Cela permet d'interpréter la "confiance" du modèle dans ses prédictions.

7. **Gestion des dépassements numériques**:
   Les scores sont écrêtés pour éviter les dépassements lors du calcul de l'exponentielle dans la fonction sigmoïde. Cela garantit la stabilité numérique sans affecter significativement les probabilités estimées.

### 3.3 Pourquoi utiliser un Perceptron ?

1. **Simplicité**: Le Perceptron est l'un des modèles d'apprentissage automatique les plus simples, ce qui le rend facile à comprendre et à implémenter.

2. **Interprétabilité**: Les poids appris par le Perceptron peuvent être directement interprétés comme l'importance relative de chaque caractéristique.

3. **Efficacité computationnelle**: Le Perceptron est rapide à entraîner et à utiliser pour les prédictions, ce qui le rend adapté aux grands ensembles de données ou aux applications en temps réel.

4. **Base pour des modèles plus complexes**: Comprendre le Perceptron est crucial pour appréhender des modèles plus avancés comme les réseaux de neurones multicouches.

5. **Garantie de convergence**: Pour les problèmes linéairement séparables, le Perceptron est garanti de converger vers une solution en un nombre fini d'itérations.

### 3.4 Limites et considérations

1. **Problèmes non linéairement séparables**: Le Perceptron ne peut pas résoudre des problèmes qui ne sont pas linéairement séparables, comme le problème XOR.

2. **Sensibilité à l'ordre des données**: L'ordre dans lequel les exemples sont présentés peut affecter la solution finale.

3. **Choix des hyperparamètres**: Le taux d'apprentissage et le nombre d'itérations peuvent grandement influencer les performances et doivent être choisis avec soin.

En comprenant ces aspects du Perceptron, on peut mieux apprécier son rôle dans l'histoire de l'apprentissage automatique et son utilité continue dans certains scénarios, tout en reconnaissant ses limites qui ont motivé le développement de modèles plus avancés.






### 3.5 Explication du code

Le code du Perceptron est structuré en une classe `PerceptronClassifier` avec plusieurs méthodes clés :

1. **Initialisation (`__init__`)**:
   - Définit les hyperparamètres : nombre d'itérations et taux d'apprentissage.
   - Initialise les attributs pour les poids (w) et le biais (b) à None.

2. **Entraînement (`fit`)**:
   - Valide les données d'entrée pour s'assurer qu'elles sont appropriées.
   - Initialise les poids à zéro et le biais à zéro.
   - Convertit les étiquettes de 0/1 à -1/1 pour simplifier la règle de mise à jour.
   - Effectue plusieurs passages sur l'ensemble de données :
     - Pour chaque exemple, calcule la prédiction.
     - Si la prédiction est incorrecte, met à jour les poids et le biais.
   - Retourne l'instance de la classe pour permettre le chaînage de méthodes.

3. **Prédiction (`predict`)**:
   - Calcule le produit scalaire entre les caractéristiques et les poids, ajoute le biais.
   - Applique une fonction de seuil pour obtenir des prédictions binaires (0 ou 1).

4. **Estimation des probabilités (`predict_proba`)**:
   - Calcule les scores comme dans `predict`.
   - Applique une fonction sigmoïde pour convertir les scores en probabilités.
   - Gère les problèmes numériques en écrêtant les scores.
   - Retourne les probabilités pour les deux classes.

5. **Script principal**:
   - Génère un jeu de données synthétique.
   - Divise les données en ensembles d'entraînement et de test.
   - Crée et entraîne une instance du Perceptron.
   - Évalue le modèle en calculant la précision et la matrice de confusion.
   - Affiche les résultats et trace la frontière de décision.

Le code suit une structure logique qui reflète le processus d'apprentissage automatique :
1. Préparation des données
2. Initialisation du modèle
3. Entraînement
4. Évaluation
5. Visualisation des résultats

Cette structure est commune à de nombreux algorithmes d'apprentissage automatique, ce qui rend ce code du Perceptron un excellent exemple pour comprendre les bases de l'implémentation d'un classificateur.
