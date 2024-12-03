

# Analyse des Features Finales

## 1. Features Actuelles (Statistiques)
Pour chaque série de 512 points, nous avons :
- Moyenne, Écart-type, Min, Max
- Skewness, Kurtosis
- Quartiles (Q1, Q2, Q3)
- IQR
- Corrélation
- % données valides
≈ 13 features par série

## 2. Nouvelles Features

### A. Temporelles
**Tendances** (par série) :
- Pente globale
- Nombre de points de changement
- Moyenne des pentes locales
- Variance des pentes locales
≈ 4 features

**Cycliques** (par série) :
- Coefficient d'autocorrélation max
- Lag de l'autocorrélation max
- Période dominante
- Force de la périodicité
≈ 4 features

### B. Fréquentielles
**Spectrales** (par série) :
- Top 3 fréquences dominantes
- Amplitudes correspondantes
- Ratios entre bandes (basse/moyenne/haute)
≈ 8 features

**Énergétiques** (par série) :
- Énergie totale
- Distribution par bande (3 bandes)
- Centroïde spectral
≈ 5 features

## 3. Calcul de Dimension

### Données Originales
- 31 capteurs (2 à 32)
- 512 points par capteur
= 15,872 points par échantillon

### Features Extraites
Pour chaque capteur :
- 13 stats + 4 tendances + 4 cycliques + 8 spectrales + 5 énergétiques
= 34 features par capteur

Total : 34 × 31 = 1,054 features par échantillon

## 4. Réduction de Dimension
- Données originales : 15,872 dimensions
- Features extraites : 1,054 dimensions
- Taux de réduction : ≈ 93.4%

## 5. Regroupement Logique des Features

### Par Localisation
1. **Heart Rate** (34 features)
   - Capture rythme cardiaque et variations

2. **Hand/Chest/Foot** (chacun 272 features)
   - Temperature (34)
   - Acceleration (3 × 34 = 102)
   - Gyroscope (3 × 34 = 102)
   - Magnetometer (3 × 34 = 102)

### Par Type de Mesure
1. **Mouvement** (Acc + Gyro)
   - 612 features
   - Patterns de mouvement physique

2. **Environnement** (Temp + Mag)
   - 408 features
   - Contexte et orientation

3. **Physiologique** (Heart)
   - 34 features
   - État physiologique

## 6. Considérations

### Avantages
1. **Interprétabilité** :
   - Features avec signification physique
   - Groupement logique par type/location

2. **Efficacité** :
   - Réduction significative de dimension
   - Conservation de l'information pertinente

### Défis
1. **Sélection de Features** :
   - Possible redondance entre features
   - Nécessité d'analyse de corrélation

2. **Équilibre** :
   - Entre nombre de features et information
   - Entre types de features (temporel vs fréquentiel)

### Recommandations
1. **Post-traitement** :
   - Analyse en composantes principales (PCA)
   - Sélection de features par importance

2. **Validation** :
   - Tests de significativité par feature
   - Évaluation par type d'activité
