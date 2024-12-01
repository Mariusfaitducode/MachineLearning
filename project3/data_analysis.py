import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from toy_script import load_data

class DataAnalyzer:
    def __init__(self, X_train, y_train, X_test, subject_ids_train, subject_ids_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.subject_ids_train = subject_ids_train
        self.subject_ids_test = subject_ids_test
        
        # Définition correcte des activités selon le rapport
        self.activity_names = {
            1: 'Lying', 2: 'Sitting', 3: 'Standing', 4: 'Walking very slow',
            5: 'Normal walking', 6: 'Nordic walking', 7: 'Running', 
            8: 'Ascending stairs', 9: 'Descending stairs', 10: 'Cycling',
            11: 'Ironing', 12: 'Vacuum cleaning', 13: 'Rope jumping', 
            14: 'Playing soccer'
        }
        
        # Définition des groupes de capteurs
        self.sensor_groups = {
            'Heart': [2],
            'Hand': {
                'Temperature': [3],
                'Acceleration': [4, 5, 6],
                'Gyroscope': [7, 8, 9],
                'Magnetometer': [10, 11, 12]
            },
            'Chest': {
                'Temperature': [13],
                'Acceleration': [14, 15, 16],
                'Gyroscope': [17, 18, 19],
                'Magnetometer': [20, 21, 22]
            },
            'Foot': {
                'Temperature': [23],
                'Acceleration': [24, 25, 26],
                'Gyroscope': [27, 28, 29],
                'Magnetometer': [30, 31, 32]
            }
        }
        
        os.makedirs('project3/data_analysis', exist_ok=True)


    def plot_activity_distribution(self):
        """
        Visualise la distribution des activités
        """
        plt.figure(figsize=(12, 6))
        activity_counts = pd.Series(self.y_train).map(self.activity_names).value_counts()
        sns.barplot(x=activity_counts.values, y=activity_counts.index)
        plt.title('Distribution des activités')
        plt.xlabel('Nombre d\'échantillons')
        plt.tight_layout()
        plt.savefig('project3/data_analysis/activity_distribution.png')
        plt.close()

    def plot_sensor_patterns(self, activity_id=1):
        """
        Visualise tous les patterns des capteurs pour une activité donnée
        avec zones grisées pour les données manquantes
        """
        activity_name = self.activity_names[activity_id]
        activity_samples = self.X_train[self.y_train == activity_id]
        sample = activity_samples[0]
        
        fig, axes = plt.subplots(5, 3, figsize=(20, 25))
        
        def plot_with_missing_zones(ax, data, label=None, color=None):
            """Helper function pour tracer les données avec zones manquantes"""
            missing_mask = data == -999999.99
            valid_data = data.copy()
            valid_data[missing_mask] = np.nan
            
            # Tracer les données valides
            line = ax.plot(valid_data, label=label, color=color)
            
            # Ajouter les zones grisées pour les données manquantes
            missing_ranges = []
            start = None
            for i, missing in enumerate(missing_mask):
                if missing and start is None:
                    start = i
                elif not missing and start is not None:
                    missing_ranges.append((start, i))
                    start = None
            if start is not None:
                missing_ranges.append((start, len(missing_mask)))
                
            for start, end in missing_ranges:
                ax.axvspan(start, end, color='gray', alpha=0.2)
            
            return line[0].get_color() if line else None
        
        # 1. Heart Rate
        heart_data = sample[0:512]
        plot_with_missing_zones(axes[0, 0], heart_data, label='BPM', color='red')
        axes[0, 0].set_title('Heart Rate (BPM)')
        axes[0, 0].legend()
        
        # Statistiques du rythme cardiaque
        valid_heart_data = heart_data[heart_data != -999999.99]
        if len(valid_heart_data) > 0:
            stats_text = f'Moyenne: {np.mean(valid_heart_data):.1f} BPM\n'
            stats_text += f'Max: {np.max(valid_heart_data):.1f} BPM\n'
            stats_text += f'Min: {np.min(valid_heart_data):.1f} BPM\n'
            stats_text += f'Écart-type: {np.std(valid_heart_data):.1f} BPM\n'
            stats_text += f'Données manquantes: {np.sum(heart_data == -999999.99)/len(heart_data):.1%}'
        else:
            stats_text = "Données non disponibles"
        axes[0, 1].text(0.1, 0.5, stats_text, fontsize=12)
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        
        # 2. Températures
        temps = {
            'Hand': sample[512:1024],
            'Chest': sample[11*512:12*512],
            'Foot': sample[21*512:22*512]
        }
        
        for idx, (location, temp_data) in enumerate(temps.items()):
            plot_with_missing_zones(axes[1, idx], temp_data, label=f'{location} Temp')
            axes[1, idx].set_title(f'{location} Temperature (°C)')
            
            valid_temp = temp_data[temp_data != -999999.99]
            if len(valid_temp) > 0:
                stats_text = f'Moyenne: {np.mean(valid_temp):.1f}°C\n'
                stats_text += f'Max: {np.max(valid_temp):.1f}°C\n'
                stats_text += f'Min: {np.min(valid_temp):.1f}°C\n'
                stats_text += f'Données manquantes: {np.sum(temp_data == -999999.99)/len(temp_data):.1%}'
            else:
                stats_text = "Données non disponibles"
            
            axes[1, idx].text(0.1, 0.9, stats_text, transform=axes[1, idx].transAxes)
            axes[1, idx].legend()
        
        # 3-5. Capteurs par partie du corps
        locations = ['Hand', 'Chest', 'Foot']
        for loc_idx, location in enumerate(locations):
            row_idx = loc_idx + 2
            
            # Accéléromètre
            start_idx = self.sensor_groups[location]['Acceleration'][0] - 2
            for axis in range(3):
                data = sample[(start_idx + axis)*512:(start_idx + axis + 1)*512]
                plot_with_missing_zones(axes[row_idx, 0], data, label=f'Axe {axis+1}')
            axes[row_idx, 0].set_title(f'{location} Acceleration (m/s²)')
            axes[row_idx, 0].legend()
            
            # Gyroscope
            start_idx = self.sensor_groups[location]['Gyroscope'][0] - 2
            for axis in range(3):
                data = sample[(start_idx + axis)*512:(start_idx + axis + 1)*512]
                plot_with_missing_zones(axes[row_idx, 1], data, label=f'Axe {axis+1}')
            axes[row_idx, 1].set_title(f'{location} Gyroscope (rad/s)')
            axes[row_idx, 1].legend()
            
            # Magnétomètre
            start_idx = self.sensor_groups[location]['Magnetometer'][0] - 2
            for axis in range(3):
                data = sample[(start_idx + axis)*512:(start_idx + axis + 1)*512]
                plot_with_missing_zones(axes[row_idx, 2], data, label=f'Axe {axis+1}')
            axes[row_idx, 2].set_title(f'{location} Magnetometer (μT)')
            axes[row_idx, 2].legend()
        
        plt.suptitle(f'Patterns des capteurs pour l\'activité: {activity_name}\n(échantillon de 5 secondes)', size=16)
        plt.tight_layout()
        plt.savefig(f'project3/data_analysis/sensor_patterns_{activity_name}.png')
        plt.close()

    def plot_all_activities_patterns(self):
        """
        Génère les visualisations pour toutes les activités
        """
        for activity_id in self.activity_names.keys():
            print(f"Génération des patterns pour l'activité: {self.activity_names[activity_id]}")
            self.plot_sensor_patterns(activity_id)

    def analyze_sensor_statistics(self):
        """
        Analyse les statistiques de chaque capteur
        """
        stats_dict = {}
        for sensor in range(31):
            start_idx = sensor * 512
            end_idx = (sensor + 1) * 512
            sensor_data = self.X_train[:, start_idx:end_idx]
            
            stats_dict[f'Capteur_{sensor+1}'] = {
                'mean': np.mean(sensor_data),
                'std': np.std(sensor_data),
                'min': np.min(sensor_data),
                'max': np.max(sensor_data),
                'missing': np.sum(sensor_data == -999999.99)
            }
        
        return pd.DataFrame(stats_dict).T
    

    def analyze_sensor_group(self, group_name, sensor_ids):
        """
        Analyse un groupe de capteurs spécifique
        """
        stats_dict = {}
        for sensor_id in sensor_ids:
            idx = sensor_id - 2  # Ajustement car les capteurs commencent à 2
            start_idx = idx * 512
            end_idx = (idx + 1) * 512
            sensor_data = self.X_train[:, start_idx:end_idx]
            
            stats_dict[f'Sensor_{sensor_id}'] = {
                'mean': np.mean(sensor_data[sensor_data != -999999.99]),
                'std': np.std(sensor_data[sensor_data != -999999.99]),
                'missing_ratio': np.sum(sensor_data == -999999.99) / sensor_data.size
            }
        
        return pd.DataFrame(stats_dict)

    def plot_sensor_correlations(self):
        """
        Visualise les corrélations entre les capteurs du même type
        """
        for location in ['Hand', 'Chest', 'Foot']:
            for sensor_type in ['Acceleration', 'Gyroscope', 'Magnetometer']:
                if sensor_type in self.sensor_groups[location]:
                    sensors = self.sensor_groups[location][sensor_type]
                    if len(sensors) > 1:
                        # Créer une matrice pour stocker les données des capteurs
                        sensor_data = []
                        for s in sensors:
                            # Extraire les données du capteur
                            data = self.X_train[:, (s-2)*512:(s-2)*512+512]
                            # Remplacer les valeurs manquantes par la moyenne du capteur
                            mask = data == -999999.99
                            data[mask] = np.mean(data[~mask])
                            # Calculer la moyenne sur l'axe temporel
                            mean_data = np.mean(data, axis=1)
                            sensor_data.append(mean_data)
                        
                        # Calculer la matrice de corrélation
                        sensor_data = np.array(sensor_data)
                        correlation_matrix = np.corrcoef(sensor_data)
                        
                        # Visualisation
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(correlation_matrix, 
                                  annot=True, 
                                  fmt='.2f',
                                  cmap='coolwarm',
                                  vmin=-1, 
                                  vmax=1,
                                  xticklabels=[f'Axe {i+1}' for i in range(len(sensors))],
                                  yticklabels=[f'Axe {i+1}' for i in range(len(sensors))])
                        plt.title(f'Corrélations {location} {sensor_type}')
                        plt.tight_layout()
                        plt.savefig(f'project3/data_analysis/{location}_{sensor_type}_correlations.png')
                        plt.close()

                        # Afficher les statistiques de corrélation
                        print(f"\n=== Corrélations {location} {sensor_type} ===")
                        for i in range(len(sensors)):
                            for j in range(i+1, len(sensors)):
                                print(f"Corrélation entre axes {i+1} et {j+1}: {correlation_matrix[i,j]:.3f}")

    def analyze_physical_measurements(self):
        """
        Analyse spécifique pour les mesures physiques (température, rythme cardiaque)
        """
        # Analyse du rythme cardiaque
        heart_rate = self.X_train[:, 0:512]  # Sensor 2
        print("\n=== Analyse du rythme cardiaque ===")
        print(f"Moyenne: {np.mean(heart_rate[heart_rate != -999999.99]):.2f} bpm")
        print(f"Max: {np.max(heart_rate[heart_rate != -999999.99]):.2f} bpm")
        print(f"Min: {np.min(heart_rate[heart_rate != -999999.99]):.2f} bpm")
        
        # Analyse des températures
        for location, sensor_id in [('Hand', 3), ('Chest', 13), ('Foot', 23)]:
            temp_data = self.X_train[:, (sensor_id-2)*512:(sensor_id-2)*512+512]
            valid_temp = temp_data[temp_data != -999999.99]
            print(f"\n=== Température {location} ===")
            print(f"Moyenne: {np.mean(valid_temp):.2f}°C")
            print(f"Max: {np.max(valid_temp):.2f}°C")
            print(f"Min: {np.min(valid_temp):.2f}°C")

    def plot_global_correlations(self):
        """
        Visualise la matrice de corrélation entre tous les capteurs
        """
        print("\nCalcul de la matrice de corrélation globale...")
        
        # Préparer les données moyennées par capteur
        sensor_means = []
        sensor_labels = []
        
        for sensor_id in range(2, 33):  # Capteurs 2 à 32
            # Extraire les données du capteur
            data = self.X_train[:, (sensor_id-2)*512:(sensor_id-1)*512]
            # Remplacer les valeurs manquantes par la moyenne du capteur
            mask = data == -999999.99
            data[mask] = np.mean(data[~mask])
            # Calculer la moyenne sur l'axe temporel
            mean_data = np.mean(data, axis=1)
            sensor_means.append(mean_data)
            
            # Créer un label approprié selon le type de capteur
            if sensor_id == 2:
                label = "Heart"
            elif sensor_id == 3:
                label = "Hand_Temp"
            elif sensor_id in [4, 5, 6]:
                label = f"Hand_Acc_{sensor_id-3}"
            elif sensor_id in [7, 8, 9]:
                label = f"Hand_Gyro_{sensor_id-6}"
            elif sensor_id in [10, 11, 12]:
                label = f"Hand_Mag_{sensor_id-9}"
            elif sensor_id == 13:
                label = "Chest_Temp"
            elif sensor_id in [14, 15, 16]:
                label = f"Chest_Acc_{sensor_id-13}"
            elif sensor_id in [17, 18, 19]:
                label = f"Chest_Gyro_{sensor_id-16}"
            elif sensor_id in [20, 21, 22]:
                label = f"Chest_Mag_{sensor_id-19}"
            elif sensor_id == 23:
                label = "Foot_Temp"
            elif sensor_id in [24, 25, 26]:
                label = f"Foot_Acc_{sensor_id-23}"
            elif sensor_id in [27, 28, 29]:
                label = f"Foot_Gyro_{sensor_id-26}"
            else:
                label = f"Foot_Mag_{sensor_id-29}"
            
            sensor_labels.append(label)
        
        # Calculer la matrice de corrélation
        sensor_data = np.array(sensor_means)
        correlation_matrix = np.corrcoef(sensor_data)
        
        # Visualisation
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.2f',
                    cmap='coolwarm',
                    vmin=-1, 
                    vmax=1,
                    xticklabels=sensor_labels,
                    yticklabels=sensor_labels)
        plt.title('Matrice de corrélation globale entre capteurs')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('project3/data_analysis/global_correlation_matrix.png')
        plt.close()
        
        # Identifier les corrélations les plus fortes (en valeur absolue)
        print("\nCorrelations les plus fortes (|r| > 0.5):")
        for i in range(len(sensor_labels)):
            for j in range(i+1, len(sensor_labels)):
                if abs(correlation_matrix[i,j]) > 0.5:
                    print(f"{sensor_labels[i]} - {sensor_labels[j]}: {correlation_matrix[i,j]:.3f}")

    def analyze_location_correlations(self, location):
        """
        Analyse les corrélations entre tous les capteurs d'une même partie du corps
        """
        print(f"\n=== Corrélations pour {location} ===")
        
        # Préparer les données
        sensor_data = []
        labels = []
        
        # Ajouter accéléromètre
        for i, sensor_id in enumerate(self.sensor_groups[location]['Acceleration']):
            data = self.X_train[:, (sensor_id-2)*512:(sensor_id-1)*512]
            mask = data == -999999.99
            data[mask] = np.mean(data[~mask])
            sensor_data.append(np.mean(data, axis=1))
            labels.append(f'Acc_{i+1}')
        
        # Ajouter gyroscope
        for i, sensor_id in enumerate(self.sensor_groups[location]['Gyroscope']):
            data = self.X_train[:, (sensor_id-2)*512:(sensor_id-1)*512]
            mask = data == -999999.99
            data[mask] = np.mean(data[~mask])
            sensor_data.append(np.mean(data, axis=1))
            labels.append(f'Gyro_{i+1}')
        
        # Ajouter magnétomètre
        for i, sensor_id in enumerate(self.sensor_groups[location]['Magnetometer']):
            data = self.X_train[:, (sensor_id-2)*512:(sensor_id-1)*512]
            mask = data == -999999.99
            data[mask] = np.mean(data[~mask])
            sensor_data.append(np.mean(data, axis=1))
            labels.append(f'Mag_{i+1}')
        
        # Calculer et afficher la matrice de corrélation
        sensor_data = np.array(sensor_data)
        correlation_matrix = np.corrcoef(sensor_data)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.2f',
                    cmap='coolwarm',
                    vmin=-1, 
                    vmax=1,
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title(f'Corrélations entre capteurs - {location}')
        plt.tight_layout()
        plt.savefig(f'project3/data_analysis/{location}_sensors_correlations.png')
        plt.close()
        
        # Afficher les corrélations significatives
        print("\nCorrélations significatives (|r| > 0.3):")
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if abs(correlation_matrix[i,j]) > 0.3:
                    print(f"{labels[i]} - {labels[j]}: {correlation_matrix[i,j]:.3f}")

    def analyze_all_locations(self):
        """
        Analyse les corrélations pour chaque partie du corps
        """
        for location in ['Hand', 'Chest', 'Foot']:
            self.analyze_location_correlations(location)

    def analyze_temperature_correlations(self):
        """
        Analyse et visualise les corrélations entre les températures des différentes parties du corps
        """
        print("\n=== Analyse des corrélations entre températures ===")
        
        # Extraire les données de température
        temp_data = {
            'Hand': self.X_train[:, 512:1024],         # Sensor 3
            'Chest': self.X_train[:, 11*512:12*512],   # Sensor 13
            'Foot': self.X_train[:, 21*512:22*512]     # Sensor 23
        }
        
        # Calculer les moyennes des températures en ignorant les valeurs manquantes
        temp_means = {}
        for location, data in temp_data.items():
            valid_mask = data != -999999.99
            temp_means[location] = np.array([
                np.mean(row[valid_mask[i]]) if np.any(valid_mask[i]) else np.nan 
                for i, row in enumerate(data)
            ])
        
        # Créer un DataFrame pour faciliter l'analyse
        df_temps = pd.DataFrame(temp_means)
        
        # Calculer la matrice de corrélation
        correlation_matrix = df_temps.corr()
        
        # Visualisation
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.3f',
                    cmap='coolwarm',
                    vmin=-1, 
                    vmax=1,
                    center=0)
        plt.title('Corrélations entre les températures')
        plt.tight_layout()
        plt.savefig('project3/data_analysis/temperature_correlations.png')
        plt.close()
        
        # Afficher les statistiques détaillées
        print("\nStatistiques des températures:")
        for location in temp_means:
            valid_temp = temp_means[location][~np.isnan(temp_means[location])]
            print(f"\n{location}:")
            print(f"Moyenne: {np.mean(valid_temp):.2f}°C")
            print(f"Écart-type: {np.std(valid_temp):.2f}°C")
            print(f"Min: {np.min(valid_temp):.2f}°C")
            print(f"Max: {np.max(valid_temp):.2f}°C")
            print(f"Données manquantes: {np.sum(np.isnan(temp_means[location]))/len(temp_means[location]):.1%}")
        
        # Afficher les corrélations
        print("\nCorrélations entre températures:")
        for i in range(len(correlation_matrix.index)):
            for j in range(i+1, len(correlation_matrix.columns)):
                loc1 = correlation_matrix.index[i]
                loc2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                print(f"{loc1} - {loc2}: {corr:.3f}")
        
        # Analyse par activité
        print("\nMoyennes des températures par activité:")
        df_temps['Activity'] = self.y_train
        activity_means = df_temps.groupby('Activity').mean()
        
        plt.figure(figsize=(15, 6))
        activity_means.plot(kind='bar')
        plt.title('Températures moyennes par activité')
        plt.xlabel('Activité')
        plt.ylabel('Température (°C)')
        plt.xticks(range(len(self.activity_names)), 
                   [self.activity_names[i+1] for i in range(len(self.activity_names))],
                   rotation=45,
                   ha='right')
        plt.legend(title='Location')
        plt.tight_layout()
        plt.savefig('project3/data_analysis/temperature_by_activity.png')
        plt.close()

    def plot_missing_data(self):
        """
        Visualise les données manquantes pour chaque capteur
        """
        print("\n=== Visualisation des données manquantes ===")
        
        # Calculer le pourcentage de données manquantes par capteur
        missing_percentages = []
        sensor_names = []
        
        # Préparer les données
        for sensor_id in range(2, 33):
            data = self.X_train[:, (sensor_id-2)*512:(sensor_id-1)*512]
            missing_percent = np.sum(data == -999999.99) / data.size * 100
            missing_percentages.append(missing_percent)
            
            # Créer un nom explicite pour le capteur
            if sensor_id == 2:
                name = "Heart Rate"
            elif sensor_id == 3:
                name = "Hand Temp"
            elif sensor_id == 13:
                name = "Chest Temp"
            elif sensor_id == 23:
                name = "Foot Temp"
            else:
                sensor_type = ""
                if sensor_id in [4, 5, 6, 14, 15, 16, 24, 25, 26]:
                    sensor_type = "Acc"
                elif sensor_id in [7, 8, 9, 17, 18, 19, 27, 28, 29]:
                    sensor_type = "Gyro"
                else:
                    sensor_type = "Mag"
                
                location = "Hand" if sensor_id < 13 else ("Chest" if sensor_id < 23 else "Foot")
                axis = (sensor_id - 4) % 3 + 1 if sensor_type == "Acc" else \
                       (sensor_id - 7) % 3 + 1 if sensor_type == "Gyro" else \
                       (sensor_id - 10) % 3 + 1
                name = f"{location} {sensor_type} {axis}"
            
            sensor_names.append(name)
        
        # Créer le graphique
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(missing_percentages)), missing_percentages)
        
        # Colorer les barres par type de capteur
        colors = {'Heart Rate': 'red', 'Temp': 'orange', 
                  'Acc': 'blue', 'Gyro': 'green', 'Mag': 'purple'}
        
        for i, bar in enumerate(bars):
            sensor_type = sensor_names[i].split()[1] if len(sensor_names[i].split()) > 1 else sensor_names[i]
            if sensor_type in colors:
                bar.set_color(colors[sensor_type])
            elif "Temp" in sensor_names[i]:
                bar.set_color(colors['Temp'])
        
        plt.xticks(range(len(sensor_names)), sensor_names, rotation=45, ha='right')
        plt.ylabel('Pourcentage de données manquantes')
        plt.title('Données manquantes par capteur')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajouter les pourcentages au-dessus des barres
        for i, v in enumerate(missing_percentages):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig('project3/data_analysis/missing_data_distribution.png')
        plt.close()

    def plot_activity_by_subject(self):
        """
        Visualise la répartition des activités par sujet
        """
        print("\n=== Visualisation des activités par sujet ===")
        
        # Créer un DataFrame avec les sujets et leurs activités
        df = pd.DataFrame({
            'Subject': self.subject_ids_train,
            'Activity': [self.activity_names[y] for y in self.y_train]
        })
        
        # Calculer la distribution des activités par sujet
        activity_counts = pd.crosstab(df['Subject'], df['Activity'])
        
        # Créer un graphique empilé
        plt.figure(figsize=(15, 8))
        activity_counts.plot(kind='bar', stacked=True)
        plt.title('Distribution des activités par sujet')
        plt.xlabel('Sujet')
        plt.ylabel('Nombre d\'échantillons')
        plt.legend(title='Activité', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('project3/data_analysis/activity_by_subject_stacked.png')
        plt.close()
        
        # Créer un heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(activity_counts, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Heatmap des activités par sujet')
        plt.ylabel('Sujet')
        plt.xlabel('Activité')
        plt.tight_layout()
        plt.savefig('project3/data_analysis/activity_by_subject_heatmap.png')
        plt.close()
        
        # Afficher les statistiques
        print("\nNombre d'échantillons par sujet:")
        print(df['Subject'].value_counts().sort_index())
        
        print("\nNombre d'activités différentes par sujet:")
        activities_per_subject = activity_counts.astype(bool).sum(axis=1)
        print(activities_per_subject)

    def run_complete_analysis(self):
        """
        Exécute l'analyse complète des données
        """
        print("=== Début de l'analyse complète des données ===")
        
        self.plot_missing_data()
        self.plot_activity_by_subject()
        
        self.plot_activity_distribution()

        for activity_id in self.activity_names.keys():
            self.plot_sensor_patterns(activity_id)
            
        # self.plot_sensor_correlations()
        self.analyze_physical_measurements()
        self.plot_global_correlations()
        self.analyze_all_locations()
        self.analyze_temperature_correlations()

def main():
    # Charger les données (à adapter selon votre structure)
    X_train, y_train, X_test, subject_ids_train, subject_ids_test = load_data()
    
    # Créer l'analyseur
    analyzer = DataAnalyzer(X_train, y_train, X_test, subject_ids_train, subject_ids_test)
    
    # Lancer l'analyse
    analyzer.run_complete_analysis()
    pass

if __name__ == "__main__":
    main()
