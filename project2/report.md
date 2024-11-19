## 1. **Question 2.1 - Difficulté d'estimation de l'erreur résiduelle**
  
L'estimation de l'erreur résiduelle dans le contexte du dataset Wine Quality présente des défis particuliers qui rendent cette tâche pratiquement impossible. Pour bien comprendre cette difficulté, il est essentiel de définir d'abord ce qu'est l'erreur résiduelle. Il s'agit de la composante irréductible de l'erreur, représentant le bruit inhérent aux données qui ne peut être expliqué même par le meilleur modèle possible. Cette erreur existe naturellement dans toutes les données du monde réel et représente la limite fondamentale de ce qu'un modèle peut accomplir.
Dans notre cas spécifique du Wine Quality dataset, la principale difficulté réside dans l'impossibilité d'accéder à la "vraie" fonction sous-jacente qui détermine la qualité d'un vin. En effet, pour calculer précisément l'erreur résiduelle, nous aurions besoin de connaître la fonction idéale f(x) qui, pour chaque ensemble de caractéristiques d'un vin, donnerait sa "vraie" qualité intrinsèque. Cependant, cette fonction idéale n'existe pas dans la réalité, car la notation de la qualité d'un vin comporte une part inhérente de subjectivité et de variabilité.
Un aspect crucial qui complique davantage cette estimation est la nature même des données dont nous disposons. Dans notre dataset, chaque vin n'est évalué qu'une seule fois, ce qui signifie que pour une combinaison donnée de caractéristiques (acidité, teneur en alcool, etc.), nous n'avons qu'une seule note de qualité. Cette limitation est fondamentale car, pour véritablement estimer l'erreur résiduelle, nous aurions besoin de multiples évaluations du même vin exact par différents experts. Ces évaluations répétées nous permettraient d'observer la variabilité naturelle dans les notations et ainsi de distinguer le bruit aléatoire de la relation sous-jacente entre les caractéristiques du vin et sa qualité.


## 2. **Question 2.2 - Protocole d'estimation**

Pour estimer la variance, l'erreur attendue et la somme du biais et de l'erreur résiduelle à partir du pool de données P, nous proposons un protocole en quatre étapes principales. Ce protocole est conçu pour obtenir des estimations fiables tout en tenant compte des contraintes inhérentes au problème.

**Étape 1 : Préparation initiale des données**

Cette première étape est cruciale pour garantir la validité de nos estimations :
- Division du jeu de données complet (NS = 6497 échantillons) en :
  * Un pool P pour l'apprentissage (80% des données)
  * Un ensemble de test fixe (20% des données)

*Justification* :
- La séparation train/test est essentielle pour évaluer la véritable capacité de généralisation des modèles
- L'ensemble de test doit rester fixe pour assurer la cohérence des comparaisons entre différents modèles
- La proportion 80/20 est un standard qui offre un bon compromis entre la taille du pool d'apprentissage et la fiabilité des estimations sur l'ensemble de test

**Étape 2 : Création des échantillons d'apprentissage**

Cette étape vise à simuler la variabilité naturelle dans l'apprentissage :
- À partir du pool P, nous générons B = 100 échantillons d'apprentissage différents
- Chaque échantillon contient N = 250 observations
- La sélection est faite aléatoirement sans remise pour chaque échantillon

*Justification* :
- B = 100 itérations offrent un bon compromis entre précision statistique et temps de calcul
- N = 250 est suffisamment grand pour l'apprentissage mais reste petit par rapport à NS
- L'échantillonnage sans remise évite les duplications au sein d'un même échantillon
- La répétition du processus B fois permet de capturer la variabilité due à l'échantillonnage

**Étape 3 : Entraînement et prédiction**

Pour chaque échantillon d'apprentissage i de 1 à B :
1. Entraînement d'un nouveau modèle fi sur l'échantillon i
2. Prédictions fi(x) sur l'ensemble de test fixe
3. Stockage des prédictions dans une matrice de dimension B × M (M étant la taille de l'ensemble de test)

*Justification* :
- L'utilisation d'un nouveau modèle à chaque itération assure l'indépendance des apprentissages
- Les prédictions sur l'ensemble de test fixe permettent une comparaison cohérente
- Le stockage des prédictions permet un calcul efficace des métriques finales

**Étape 4 : Calcul des estimations**

Cette étape finale permet d'obtenir nos trois métriques d'intérêt :

1. **Variance** :
```python
variance = moyenne(variance_des_predictions_pour_chaque_point)
```
*Justification* : La variance mesure la dispersion des prédictions pour un même point, moyennée sur tous les points de test. Elle quantifie l'instabilité du modèle face à différents échantillons d'apprentissage.

2. **Erreur attendue** :
```python
erreur_attendue = moyenne((predictions - vraies_valeurs)²)
```
*Justification* : L'erreur quadratique moyenne mesure la performance globale du modèle, prenant en compte à la fois le biais et la variance.

3. **Biais² + Erreur résiduelle** :
```python
biais_plus_residuel = erreur_attendue - variance
```
*Justification* : Bien que nous ne puissions pas séparer le biais de l'erreur résiduelle, leur somme nous permet d'évaluer la part de l'erreur qui n'est pas due à la variance.

**Avantages du protocole :**

1. **Robustesse** :
   - Les estimations sont moyennées sur de nombreux échantillons
   - L'utilisation d'un ensemble de test fixe assure la cohérence des comparaisons

2. **Flexibilité** :
   - Le protocole s'adapte à différents types de modèles
   - Les paramètres B et N peuvent être ajustés selon les besoins

3. **Efficacité** :
   - Permet d'estimer les composantes clés de l'erreur
   - Facilite l'analyse du compromis biais-variance

4. **Praticité** :
   - Simple à implémenter
   - Résultats faciles à interpréter

Ce protocole, bien que ne permettant pas d'isoler l'erreur résiduelle, nous fournit toutes les informations nécessaires pour analyser comment les hyperparamètres des différentes méthodes affectent le compromis biais-variance, ce qui est l'objectif principal de notre étude.


## 3. **Question 2.3 - Implémentation et analyse**

**3.1 Implémentation du protocole**

Notre implémentation du protocole d'estimation se compose de deux fichiers principaux :

**bias_variance.py** : Contient les fonctions fondamentales d'estimation
```python
def estimate_bias_variance(X, y, estimator, n_samples=250, n_iterations=100):
    """
    Implémente le protocole d'estimation en quatre étapes :
    1. Séparation train/test (80/20)
    2. Génération de B=100 échantillons de taille N=250
    3. Entraînement des modèles et prédictions
    4. Calcul des métriques (variance, erreur attendue, biais²+résiduel)
    """
```

```python
def evaluate_model_complexity(X, y, create_model, param_range):
    """
    Évalue l'impact des hyperparamètres sur le compromis biais-variance :
    - Teste différentes valeurs d'hyperparamètres
    - Applique estimate_bias_variance pour chaque configuration
    - Retourne les courbes d'évolution des métriques
    """
```

**main.py** : Applique le protocole aux trois modèles demandés
1. k-NN : variation du nombre de voisins k
2. Lasso : variation du paramètre de régularisation λ
3. Arbres de décision : variation de la profondeur maximale

Les hyperparamètres ont été choisis pour couvrir un large spectre de complexité :
- k-NN : k ∈ [1, 3, 5, 7, 11, 15, 21, 31, 51]
- Lasso : λ ∈ [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- Arbres : profondeur ∈ [1, 2, 3, 5, 7, 10, 15, 20, None]

Pour chaque configuration, le code :
1. Crée le modèle avec les hyperparamètres spécifiés
2. Applique le protocole d'estimation
3. Calcule et stocke les trois métriques
4. Génère des visualisations du compromis biais-variance

Cette implémentation respecte fidèlement le protocole défini précédemment tout en offrant la flexibilité nécessaire pour l'analyse comparative des différentes méthodes d'apprentissage.


## 4. **Question 2.4 - Sample size analyse**

**Mise en place du test**

Pour étudier l'impact de la taille d'échantillon sur le compromis biais-variance, nous avons adapté notre protocole initial avec les modifications suivantes :

1. **Sélection des tailles d'échantillon** :
   - Nous testons une gamme de tailles : [50, 100, 250, 500, 1000, 2000]
   - Cette progression quasi-géométrique permet d'observer l'évolution sur différents ordres de grandeur
   - La limite supérieure (2000) reste inférieure à la taille totale du dataset (6497)

2. **Configuration des modèles** :
   - k-NN : k fixé à 15 (valeur optimale identifiée précédemment)
   - Lasso : λ fixé à 0.01 (valeur optimale identifiée précédemment)
   - Arbres de décision : deux configurations testées
     * Profondeur fixe (3) pour un modèle contraint
     * Profondeur non limitée (None) pour un modèle flexible

3. **Processus d'évaluation** :
   - Pour chaque taille d'échantillon N :
     * Application du protocole d'estimation (B=100 itérations)
     * Calcul des trois métriques (erreur, variance, biais²+résiduel)
     * Conservation des résultats pour chaque modèle

4. **Visualisation** :
   - Graphiques en échelle logarithmique pour la taille d'échantillon
   - Courbes séparées pour chaque modèle
   - Affichage des trois métriques sur chaque graphique

Cette approche nous permet d'observer comment la taille d'échantillon influence le compromis biais-variance pour différents types de modèles, tout en maintenant les autres paramètres constants.