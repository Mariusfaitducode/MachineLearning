2.1 Pourquoi est-il difficile d'estimer l'erreur résiduelle ?
L'erreur résiduelle, notée 
𝜎
2
σ 
2
 , correspond au bruit irréductible dans les données, lié à des facteurs aléatoires ou non mesurés. Dans notre contexte, plusieurs éléments rendent son estimation difficile :

Subjectivité des évaluations : Le dataset Wine Quality repose sur des jugements humains pour noter la qualité des vins. Ces notes sont influencées par les préférences et perceptions personnelles des dégustateurs, ce qui ajoute un bruit important et imprévisible aux données.

Inconnaissance de 
𝑓
(
𝑥
)
f(x) : La fonction idéale 
𝑓
(
𝑥
)
f(x), qui lie les caractéristiques des vins à leur qualité, est inconnue. Cela nous empêche de séparer précisément 
𝜎
2
σ 
2
  des autres composantes de l’erreur.

Facteurs non mesurés : Des variables comme l’origine du vin ou les conditions de dégustation ne sont pas incluses dans les données, ce qui contribue à l'erreur résiduelle sans qu’on puisse la modéliser.

En pratique, 
𝜎
2
σ 
2
  reste constant pour un dataset donné. Plutôt que de l’estimer directement, nous nous concentrons sur la somme biais + erreur résiduelle, suffisante pour comparer les performances des modèles.


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


Résultats :

=== k-NN Analysis ===
k (n_neighbors)=1: Error=0.9584, Variance=0.4810, Bias²+Residual=0.4774
k (n_neighbors)=3: Error=0.6607, Variance=0.1671, Bias²+Residual=0.4936
k (n_neighbors)=5: Error=0.6050, Variance=0.1021, Bias²+Residual=0.5030
k (n_neighbors)=7: Error=0.5846, Variance=0.0738, Bias²+Residual=0.5108
k (n_neighbors)=11: Error=0.5710, Variance=0.0481, Bias²+Residual=0.5229
k (n_neighbors)=15: Error=0.5684, Variance=0.0360, Bias²+Residual=0.5324
k (n_neighbors)=21: Error=0.5705, Variance=0.0262, Bias²+Residual=0.5443
k (n_neighbors)=31: Error=0.5782, Variance=0.0181, Bias²+Residual=0.5601
k (n_neighbors)=51: Error=0.5931, Variance=0.0115, Bias²+Residual=0.5816

=== Lasso Analysis ===
alpha=0.0001: Error=0.5706, Variance=0.0302, Bias²+Residual=0.5404
alpha=0.001: Error=0.5696, Variance=0.0294, Bias²+Residual=0.5403
alpha=0.01: Error=0.5642, Variance=0.0232, Bias²+Residual=0.5410
alpha=0.1: Error=0.5805, Variance=0.0086, Bias²+Residual=0.5719
alpha=1.0: Error=0.7379, Variance=0.0036, Bias²+Residual=0.7342
alpha=10.0: Error=0.7379, Variance=0.0036, Bias²+Residual=0.7342
alpha=100.0: Error=0.7379, Variance=0.0036, Bias²+Residual=0.7342

=== Decision Trees Analysis ===
max_depth=1: Error=0.6561, Variance=0.0451, Bias²+Residual=0.6110
max_depth=2: Error=0.6394, Variance=0.0868, Bias²+Residual=0.5526
max_depth=3: Error=0.6773, Variance=0.1447, Bias²+Residual=0.5326
max_depth=5: Error=0.7951, Variance=0.2896, Bias²+Residual=0.5055
max_depth=7: Error=0.9083, Variance=0.4185, Bias²+Residual=0.4898
max_depth=10: Error=1.0100, Variance=0.5276, Bias²+Residual=0.4825
max_depth=15: Error=1.0527, Variance=0.5679, Bias²+Residual=0.4848
max_depth=20: Error=1.0526, Variance=0.5695, Bias²+Residual=0.4831
max_depth=None: Error=1.0526, Variance=0.5695, Bias²+Residual=0.4831


**3.2 Analyse des résultats**

Les résultats obtenus pour les trois modèles montrent des comportements distincts et cohérents avec la théorie du compromis biais-variance.

**1. k-NN Analysis**

Observations clés :
- Pour k=1 : Error=0.9584, Variance=0.4810, Bias²+Residual=0.4774
- Pour k=15 : Error=0.5684, Variance=0.0360, Bias²+Residual=0.5324
- Pour k=51 : Error=0.5931, Variance=0.0115, Bias²+Residual=0.5816

Analyse :
- La variance diminue de manière monotone avec l'augmentation de k (de 0.4810 à 0.0115)
- Le biais² + résiduel augmente progressivement avec k (de 0.4774 à 0.5816)
- L'erreur totale atteint son minimum autour de k=15 (0.5684)
- Ces résultats sont cohérents avec la théorie :
  * k faible : modèle flexible → haute variance, bas biais
  * k élevé : modèle rigide → basse variance, haut biais
  * Le k optimal (≈15) représente le meilleur compromis

**2. Lasso Analysis**

Observations clés :
- Pour α=0.0001 : Error=0.5706, Variance=0.0302, Bias²+Residual=0.5404
- Pour α=0.01 : Error=0.5642, Variance=0.0232, Bias²+Residual=0.5410
- Pour α=100.0 : Error=0.7379, Variance=0.0036, Bias²+Residual=0.7342

Analyse :
- La variance diminue avec l'augmentation de α
- Le biais augmente significativement pour α > 0.1
- L'erreur minimale est atteinte pour α=0.01
- Comportement notable :
  * Stabilisation des métriques pour α ≥ 1.0
  * Faible impact sur la variance pour α très petit
  * Le compromis optimal est plus sensible que pour k-NN

**3. Decision Trees Analysis**

Observations clés :
- Pour depth=1 : Error=0.6561, Variance=0.0451, Bias²+Residual=0.6110
- Pour depth=3 : Error=0.6773, Variance=0.1447, Bias²+Residual=0.5326
- Pour depth=None : Error=1.0526, Variance=0.5695, Bias²+Residual=0.4831

Analyse :
- Augmentation dramatique de la variance avec la profondeur
- Diminution du biais avec la profondeur
- Comportement le plus extrême des trois modèles :
  * La variance passe de 0.0451 à 0.5695
  * L'erreur totale se dégrade significativement
  * Signe clair de surapprentissage pour les grandes profondeurs

**Comparaison globale**

1. **Stabilité** :
   - k-NN : le plus stable dans l'évolution des métriques
   - Lasso : transition brusque autour de α=0.1
   - Arbres : évolution la plus instable

2. **Meilleure performance** :
   - k-NN : 0.5684 (k=15)
   - Lasso : 0.5642 (α=0.01)
   - Arbres : 0.6394 (depth=2)

3. **Compromis biais-variance** :
   - k-NN offre le compromis le plus équilibré
   - Lasso maintient une variance faible
   - Arbres montrent la plus grande sensibilité à la complexité

Ces résultats confirment les principes théoriques du compromis biais-variance et montrent l'importance du choix des hyperparamètres pour chaque modèle.


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

Resultats :

Résultats pour knn:
  Taille échantillon | Erreur    | Variance  | Biais² + Résiduel
  --------------------|-----------|-----------|------------------
                50 |    0.6481 |    0.0439 |    0.6042
               100 |    0.6101 |    0.0409 |    0.5693
               250 |    0.5684 |    0.0360 |    0.5324
               500 |    0.5464 |    0.0337 |    0.5128
              1000 |    0.5265 |    0.0295 |    0.4970
              2000 |    0.5090 |    0.0233 |    0.4857

Résultats pour lasso:
  Taille échantillon | Erreur    | Variance  | Biais² + Résiduel
  --------------------|-----------|-----------|------------------
                50 |    0.6970 |    0.1567 |    0.5402
               100 |    0.6024 |    0.0603 |    0.5421
               250 |    0.5642 |    0.0232 |    0.5410
               500 |    0.5511 |    0.0101 |    0.5410
              1000 |    0.5460 |    0.0044 |    0.5417
              2000 |    0.5435 |    0.0018 |    0.5417

Résultats pour tree_fixed:
  Taille échantillon | Erreur    | Variance  | Biais² + Résiduel
  --------------------|-----------|-----------|------------------
                50 |    0.9489 |    0.4137 |    0.5352
               100 |    0.7929 |    0.2615 |    0.5315
               250 |    0.6773 |    0.1447 |    0.5326
               500 |    0.6195 |    0.0874 |    0.5322
              1000 |    0.5941 |    0.0541 |    0.5401
              2000 |    0.5805 |    0.0317 |    0.5488

Résultats pour tree_full:
  Taille échantillon | Erreur    | Variance  | Biais² + Résiduel
  --------------------|-----------|-----------|------------------
                50 |    1.1918 |    0.6674 |    0.5244
               100 |    1.0940 |    0.5892 |    0.5048
               250 |    1.0526 |    0.5695 |    0.4831
               500 |    1.0083 |    0.5517 |    0.4567
              1000 |    0.9667 |    0.5293 |    0.4374
              2000 |    0.8832 |    0.4794 |    0.4038

**4.2 Analyse des résultats**

Les résultats montrent clairement l'impact de la taille d'échantillon sur le compromis biais-variance pour les différents modèles.

**1. k-NN (k=15)**

Observations clés :
- N=50 : Error=0.6481, Variance=0.0439, Bias²+Residual=0.6042
- N=2000 : Error=0.5090, Variance=0.0233, Bias²+Residual=0.4857

Analyse :
- Diminution régulière de la variance (-47% de N=50 à N=2000)
- Réduction progressive du biais²+résiduel (-20%)
- Amélioration constante de l'erreur totale
- Comportement stable et prévisible :
  * La variance diminue de manière quasi-linéaire avec log(N)
  * Le biais diminue plus lentement mais régulièrement
  * Modèle le plus robuste aux variations de taille d'échantillon

**2. Lasso (α=0.01)**

Observations clés :
- N=50 : Error=0.6970, Variance=0.1567, Bias²+Residual=0.5402
- N=2000 : Error=0.5435, Variance=0.0018, Bias²+Residual=0.5417

Analyse :
- Réduction drastique de la variance (-99% de N=50 à N=2000)
- Biais²+résiduel remarquablement stable (≈0.54)
- Convergence rapide :
  * La variance chute rapidement jusqu'à N=500
  * Stabilisation presque complète après N=1000
  * Meilleure performance finale (0.5435)

**3. Arbres de décision**

a) **Profondeur fixe (depth=3)**
- N=50 : Error=0.9489, Variance=0.4137, Bias²+Residual=0.5352
- N=2000 : Error=0.5805, Variance=0.0317, Bias²+Residual=0.5488

Analyse :
- Forte réduction de la variance (-92%)
- Biais²+résiduel stable
- Amélioration significative de l'erreur totale (-39%)

b) **Profondeur non limitée**
- N=50 : Error=1.1918, Variance=0.6674, Bias²+Residual=0.5244
- N=2000 : Error=0.8832, Variance=0.4794, Bias²+Residual=0.4038

Analyse :
- Variance toujours élevée même avec N grand
- Diminution notable du biais²+résiduel (-23%)
- Surapprentissage persistant :
  * La variance reste élevée même avec N=2000
  * L'erreur totale reste la plus élevée des quatre configurations

**Comparaison globale**

1. **Impact sur la variance** :
   - Tous les modèles montrent une diminution de la variance avec N
   - Lasso : réduction la plus spectaculaire (-99%)
   - Arbres non limités : réduction la plus faible (-28%)

2. **Stabilité du biais** :
   - Lasso : biais le plus stable
   - k-NN : légère amélioration avec N
   - Arbres non limités : amélioration significative mais insuffisante

3. **Efficacité de l'apprentissage** :
   - Lasso : meilleure performance finale (0.5435)
   - k-NN : bon compromis (0.5090)
   - Arbres fixes : amélioration notable (0.5805)
   - Arbres non limités : toujours problématiques (0.8832)

Ces résultats confirment plusieurs principes théoriques :
1. La variance diminue généralement avec 1/N
2. Le biais est moins sensible à la taille d'échantillon
3. Les modèles plus complexes nécessitent plus de données
4. La régularisation (Lasso, k-NN) améliore la stabilité

Cette analyse démontre l'importance cruciale de la taille d'échantillon dans le compromis biais-variance, particulièrement pour les modèles complexes ou non régularisés.