# Rapport d'Analyse et de Développement : Classification de Séries Temporelles

## 1. Introduction

Ce projet avait pour objectif de développer un système de classification de séries temporelles en utilisant des techniques d'apprentissage automatique. L'approche adoptée s'est structurée en trois phases principales : l'analyse exploratoire des données, l'extraction et la transformation des caractéristiques, et le développement d'un modèle de classification.

## 2. Analyse Exploratoire des Données

### 2.1 Structure des Données

Les données utilisées dans ce projet se présentaient sous la forme de séries temporelles brutes, comprenant 512 points temporels pour chacun des 13 capteurs mesurant différentes variables physiologiques. Ces capteurs étaient répartis sur plusieurs parties du corps, incluant le buste, les jambes et les mains, et mesuraient des signaux tels que la température corporelle, l'accélération, les rotations enregistrées par des gyroscopes et les variations de champ magnétique captées par des magnétomètres, nous avions également une série temporelle pour les battements du coeur. 

La diversité des capteurs et des signaux mesurés constituait une richesse pour l’analyse, mais également un défi. Ces séries temporelles présentaient des caractéristiques variées, notamment des échelles de valeurs différentes, des niveaux de bruit distincts, das valeurs manquantes et des corrélations potentielles entre capteurs qu’il était crucial d’identifier pour mieux orienter les étapes de traitement des données.

### 2.2 Objectifs et Approche

L’objectif principal de cette étape était de comprendre les caractéristiques fondamentales des données. Pour ce faire, plusieurs analyses exploratoires ont été menées. La plus importantes étant la visualisation des séries temporelles par activité. Cela a permis d'observer les différences selon les capteurs, et de mieux comprendre les patterns caractéristiques de chaque activité. 
Des matrices de corrélation ont également été établies, révélant des liens significatifs entre certains capteurs. 
Enfin, des visualisations spécifiques ont été réalisées pour examiner la distribution et la localisation des valeurs manquantes, mettant en évidence des zones nécessitant un nettoyage important.

### 2.3 Résultats et Implications

Cette exploration a permis de clarifier plusieurs points cruciaux. Les graphiques des séries temporelles ont montré des différences notables entre les activités, confirmant que les signaux contiennent des informations pertinentes pour la classification.

Ces résultats ont également souligné l’importance d’un prétraitement robuste des données, notamment pour réduire la trop grande quantité de données. En effet, les séries temporelles brutes contenaient 13 capteurs * 512 points temporels, soit 6656 données par série. 


## 3. Transformation des Données  

La transformation des données a constitué l’étape centrale de ce projet, avec un accent particulier sur l’extraction de caractéristiques pertinentes. L’objectif principal était de réduire drastiquement la quantité de données contenues dans les séries temporelles brutes tout en conservant les informations les plus représentatives. Cette réduction était essentielle pour simplifier la complexité du problème, tout en améliorant la capacité des algorithmes de Machine Learning à exploiter les données.  

### 3.1 Extraction des Caractéristiques  

L’extraction des caractéristiques s’est appuyée sur plusieurs types de transformations pour résumer les séries temporelles en un ensemble limité, mais riche, de variables clés :  

1. **Caractéristiques statistiques globales**  
   Les premières caractéristiques extraites visaient à capturer les propriétés globales des séries temporelles. Pour chaque série, nous avons calculé des métriques telles que la moyenne, l’écart type, la médiane, le maximum, le minimum, ainsi que le premier et le troisième quartiles. Ces mesures fournissent un résumé simple mais efficace des valeurs centrales et de la dispersion.  

2. **Caractéristiques représentant les tendances locales**  
   Pour analyser les variations locales, les séries temporelles ont été découpées en 50 segments égaux. Dans chaque segment, nous avons calculé la pente de la régression linéaire, permettant ainsi d’obtenir des informations sur les tendances dans le temps. Ces pentes segmentaires ont ensuite été agrégées pour produire des mesures globales : la pente moyenne de la série, sa variation, et la dispersion des pentes. Ces données permettent de caractériser les changements graduels dans les séries.  

3. **Caractéristiques périodiques**  
   L’analyse des aspects périodiques des séries temporelles a été réalisée en identifiant les pics présents dans les données. Ces pics révèlent des structures répétitives ou des comportements oscillatoires, qui peuvent être caractéristiques de certaines activités.  

4. **Caractéristiques fréquentielles**  
   L’analyse fréquentielle, effectuée principalement via la transformation de Fourier, a permis de capturer les propriétés du spectre des fréquences. Parmi les caractéristiques extraites figuraient :  
   - La fréquence principale et son amplitude,  
   - Le centre de masse spectral,  
   - La dispersion des fréquences,  
   - Le point de coupure spectral (fréquence en dessous de laquelle une proportion significative de l’énergie est concentrée),  
   - Le flux spectral, indiquant les variations dans le spectre fréquentiel.  

5. **Caractéristiques liées à la forme des séries temporelles**  
   Enfin, des caractéristiques liées à la morphologie des séries ont été extraites. Cela incluait la largeur moyenne des pics et le nombre de passages par zéro, reflétant des aspects structurels et dynamiques des séries temporelles.  

### 3.2 Réduction Dimensionnelle  

Grâce à ce processus d’extraction, nous avons pu réduire considérablement la taille des données. Chaque série temporelle, initialement composée de 512 points, a été condensée en seulement 29 caractéristiques. Étant donné que les données comprenaient 13 capteurs, cela représentait une réduction de **512 × 13 = 6656** à **29 × 13 = 377** valeurs par observation. Cette compression a rendu le problème beaucoup plus abordable pour les algorithmes de Machine Learning, tout en concentrant davantage d’informations pertinentes dans les données.  

### 3.3 Visualisations et Validation  

Pour garantir la pertinence des caractéristiques extraites, de nombreuses visualisations ont été générées. Ces graphiques ont permis de vérifier que les transformations capturent efficacement les dynamiques importantes des séries temporelles. Par exemple, l’analyse des fréquences principales ou des tendances locales a confirmé que les données transformées présentaient des différences significatives en fonction des activités.  

### 3.4 Prétraitement et Choix Méthodologique  

Une phase de prétraitement des données avait également été envisagée. Cette étape incluait la normalisation des séries temporelles, le filtrage du bruit et l’élimination des outliers. Cependant, après expérimentation, cette approche a produit des résultats moins cohérents, probablement en raison d’une sur-correction des signaux. Finalement, cette phase de prétraitement a été écartée, et les données brutes ont été directement exploitées pour l’extraction des caractéristiques.  




## 4. Développement du Modèle  

Le développement du modèle a constitué une étape clé du projet, visant à exploiter au mieux les caractéristiques extraites pour effectuer la classification des séries temporelles. L’objectif était de sélectionner et d’optimiser un algorithme capable de gérer les relations complexes entre les caractéristiques, tout en offrant de bonnes performances en termes de précision, robustesse et généralisation.  

### 4.1 Choix de l’Algorithme  

Plusieurs algorithmes d’apprentissage supervisé ont été testés au cours de cette phase, chacun ayant ses avantages spécifiques pour des tâches de classification multiclasse. Cependant, après une analyse comparative, le **Random Forest** a été retenu comme modèle principal pour ce projet.  

Les raisons de ce choix incluaient :  
- Sa robustesse face aux données bruitées et aux outliers,  
- Sa capacité à capturer les relations non linéaires entre les caractéristiques,  
- Son aptitude à gérer un grand nombre de caractéristiques sans risque majeur de surapprentissage,  
- La possibilité d’interpréter les importances des caractéristiques, offrant une meilleure compréhension des variables les plus discriminantes.  

Outre le Random Forest, deux autres algorithmes ont également été explorés :  
- **XGBoost** (eXtreme Gradient Boosting), connu pour ses performances élevées sur les données structurées, a été testé mais n’a pas donné de meilleurs résultats. Sa complexité computationnelle plus élevée n’a pas justifié son utilisation face aux résultats satisfaisants du Random Forest.  
- **SVM** (Support Vector Machine), bien qu’efficace pour certaines tâches de classification, a montré des performances inférieures, notamment en termes de précision et de temps d’entraînement, dans le cadre de nos données multidimensionnelles.  

### 4.2 Optimisation du Modèle  

Pour garantir les meilleures performances possibles, l’optimisation du modèle Random Forest a suivi une approche méthodique :  

1. **Division des données**  
   Les données ont été divisées en deux ensembles : 80 % pour l’entraînement et 20 % pour la validation. Cette répartition a permis d’évaluer de manière fiable la capacité de généralisation du modèle.  

2. **Recherche d’hyperparamètres**  
   Une recherche par grille exhaustive a été menée pour ajuster les principaux hyperparamètres, notamment le nombre d’arbres dans la forêt (*n_estimators*), la profondeur maximale des arbres (*max_depth*), et la division des arbres (*min_samples_split*).  

3. **Validation croisée**  
   Une validation croisée stratifiée à 5 plis a été utilisée pour assurer que les performances du modèle étaient robustes et indépendantes de la répartition des données dans les ensembles d’entraînement et de validation.  

4. **Métrique de performance principale**  
   Le **F1-score pondéré** a été retenu comme métrique d’évaluation principale, en raison de son équilibre entre précision et rappel, particulièrement utile pour les données avec des classes déséquilibrées.  

### 4.3 Analyse des Caractéristiques Importantes  

L’une des forces du Random Forest réside dans sa capacité à fournir une interprétation des caractéristiques les plus importantes pour la classification. Cette analyse a révélé que :  
- Les caractéristiques **fréquentielles** étaient parmi les plus discriminantes, en particulier la fréquence principale et son amplitude.  
- Les caractéristiques liées aux **tendances locales**, telles que la pente moyenne ou la variation des pentes, contribuaient également significativement à la classification.  
- Les caractéristiques statistiques globales, bien qu’importantes pour capturer des variations de base, jouaient un rôle secondaire.  

Cette interprétation a confirmé la pertinence des choix effectués lors de la phase de transformation des données, tout en ouvrant des perspectives pour des simplifications potentielles du pipeline.  



## 5. Résultats et Performance  

### 5.1 Performances du Modèle  

Le modèle Random Forest développé pour ce projet a rapidement démontré son efficacité, atteignant dès la deuxième soumission sur Gradescope un score supérieur à **0,90**, ce qui constituait déjà un excellent résultat. Après quelques ajustements supplémentaires, et à l’issue de cinq soumissions, j’ai pu atteindre un score final de **0,94**, me plaçant en tête du leaderboard.  

Ce résultat a été confirmé par l'évaluation sur l’ensemble des données privées, où le modèle a obtenu une **accuracy finale de 0,927**, le seul à franchir la barre des **0,90**.  

### 5.2 Réflexion sur le Succès  

Je suis particulièrement fier de ces résultats, d’autant plus que j’ai réalisé l’intégralité du projet seul. Initialement, j’avais un partenaire, mais celui-ci a quitté le groupe suite à des désaccords méthodologiques. Malgré des recherches, je n’ai pas trouvé d’autres étudiants motivés pour travailler sérieusement avec moi, et j’ai donc pris la décision de poursuivre ce travail en autonomie. J’espère que cela ne posera pas de problème au regard des exigences académiques.  

### 5.3 Appréciation du Projet  

Ce projet a été pour moi une expérience particulièrement enrichissante. J’ai apprécié son aspect concret, reposant sur de véritables données issues de séries temporelles, ainsi que le défi posé par la conception d’un pipeline performant pour résoudre un problème réel. Le processus de transformation des données, en particulier, m’a passionné et m’a permis de mieux comprendre l’importance de l’extraction des caractéristiques pour simplifier et enrichir les données destinées aux modèles de Machine Learning.  
