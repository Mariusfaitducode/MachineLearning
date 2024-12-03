import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

class Visualizer:
    """Classe gérant la visualisation des données"""
    
    def __init__(self):
        self.output_dir = 'project3/analysis_results'
        os.makedirs(f'{self.output_dir}/features', exist_ok=True)
        os.makedirs(f'{self.output_dir}/correlation_matrices', exist_ok=True)
    
    def create_feature_boxplots(self, df, feature_names, location, sensor_type=None):
        """
        Crée les boxplots pour un type de capteur spécifique
        
        Args:
            df: DataFrame contenant les données
            feature_names: Liste des noms de caractéristiques
            location: Emplacement du capteur (Hand, Chest, Foot, Heart)
            sensor_type: Type de capteur (Temperature, Acceleration, etc.), None pour Heart
        """
        # Construire le préfixe pour la sélection des caractéristiques
        prefix = f"{location}_{sensor_type}" if sensor_type else f"{location}"
        sensor_features = [f for f in feature_names if f.startswith(prefix)]
        
        if not sensor_features:
            return
        
        # Définir les types de caractéristiques à visualiser
        feature_types = ['mean', 'std', 'median', 'max', 'min', 
                        'q1', 'q3', 'skew', 'kurtosis', 
                        'total_var', 'mean_var', 'stability', 'valid_ratio']
        
        # Créer la grille de subplots
        n_rows = (len(feature_types) + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        # Créer un boxplot pour chaque type de caractéristique
        for idx, feat_type in enumerate(feature_types):
            features = [f for f in sensor_features if feat_type in f]
            
            if features:
                # Préparer les données pour le boxplot
                data_melted = df.melt(id_vars='Activity', value_vars=features, 
                                      var_name='variable', value_name='value')
                
                # Créer le boxplot
                sns.boxplot(data=data_melted, x='variable', y='value', 
                            hue='Activity', ax=axes[idx])
                axes[idx].set_title(f'{feat_type}')
                
                # Configurer les labels
                n_features = len(features)
                axes[idx].set_xticks(range(n_features))
                axes[idx].set_xticklabels(features, rotation=45, ha='right')
                
                if idx >= len(feature_types) - 3:
                    axes[idx].set_xlabel('Caractéristique')
                else:
                    axes[idx].set_xlabel('')
                
                # Gérer la légende
                if idx == 0:
                    axes[idx].legend(title='Activity', 
                                   bbox_to_anchor=(1.05, 1), 
                                   loc='upper left')
                else:
                    axes[idx].get_legend().remove()
        
        # Masquer les subplots vides
        for idx in range(len(feature_types), len(axes)):
            axes[idx].set_visible(False)
        
        # Configurer le titre et sauvegarder
        title = f'Distribution des caractéristiques - {location}'
        if sensor_type:
            title += f' {sensor_type}'
        plt.suptitle(title, y=1.02, fontsize=16)
        plt.tight_layout()
        
        filename = f'{location}_features' if not sensor_type else f'{location}_{sensor_type}_features'
        plt.savefig(f'project3/analysis_results/features/{filename}.png',
                    bbox_inches='tight')
        plt.close()


    def visualize_correlation_matrices(self, df, feature_names):
        """
        Visualise les matrices de corrélation pour chaque type de capteur
        """
        # Créer le dossier de sortie
        import os
        os.makedirs('project3/analysis_results/correlation_matrices', exist_ok=True)
        
        # Pour chaque groupe de capteurs
        for location in ['Heart', 'Hand', 'Chest', 'Foot']:
            # Sélectionner les caractéristiques pour ce capteur
            location_features = [f for f in feature_names if f.startswith(location)]
            
            if location_features:
                # Créer la matrice de corrélation
                correlation_matrix = df[location_features].corr()
                
                # Visualiser la matrice de corrélation
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                            xticklabels=True, yticklabels=True)
                plt.title(f'Matrice de corrélation des caractéristiques - {location}')
                plt.tight_layout()
                plt.savefig(f'project3/analysis_results/correlation_matrices/{location}_correlation_matrix.png')
                plt.close()
    
    
    
    def visualize_features(self, transformed_data, feature_names, activity_labels):
        """Visualise les caractéristiques extraites"""
        df = pd.DataFrame(transformed_data, columns=feature_names)
        df['Activity'] = activity_labels
        
        for location in ['Heart', 'Hand', 'Chest', 'Foot']:
            sensor_types = [''] if location == 'Heart' else ['Temperature', 'Acceleration', 'Gyroscope', 'Magnetometer']
            for sensor_type in sensor_types:
                self.create_feature_boxplots(df, feature_names, location, sensor_type)
                self.create_trend_analysis_plots(df, feature_names, location, sensor_type)
                self.create_cyclic_analysis_plots(df, feature_names, location, sensor_type)
                self.create_frequency_analysis_plots(df, feature_names, location, sensor_type)
                self.create_shape_analysis_plots(df, feature_names, location, sensor_type)
                self.create_feature_comparison_plot(df, feature_names, location, sensor_type)
        
        self.visualize_correlation_matrices(df, feature_names)


    def visualize_time_series(self, data, sensor_info):
        """Visualise les séries temporelles"""
        pass
    
    def visualize_frequency_domain(self, data, sensor_info):
        """Visualise les caractéristiques fréquentielles"""
        pass

    def create_trend_analysis_plots(self, df, feature_names, location, sensor_type=None):
        """Visualise les caractéristiques de tendance"""
        # Construire le préfixe
        prefix = f"{location}_{sensor_type}" if sensor_type else f"{location}"
        trend_features = [f for f in feature_names if f.startswith(prefix) and 
                         any(tf in f for tf in ['slope', 'trend'])]
        
        if not trend_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Distribution des pentes globales
        sns.boxplot(data=df, x='Activity', y=next(f for f in trend_features if 'slope' in f), 
                    ax=axes[0])
        axes[0].set_title('Distribution des pentes par activité')
        axes[0].set_xticks(range(len(df['Activity'].unique())))
        axes[0].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 2. Relation entre pente globale et erreur
        slope_feature = next(f for f in trend_features if 'slope' in f and 'error' not in f)
        error_feature = next(f for f in trend_features if 'error' in f)
        sns.scatterplot(data=df, x=slope_feature, y=error_feature, 
                        hue='Activity', ax=axes[1])
        axes[1].set_title('Pente vs Erreur de pente')
        
        # 3. Distribution des tendances locales
        local_mean = next(f for f in trend_features if 'local_trend_mean' in f)
        local_std = next(f for f in trend_features if 'local_trend_std' in f)
        sns.boxplot(data=df, x='Activity', y=local_mean, ax=axes[2])
        axes[2].set_title('Distribution des tendances locales moyennes')
        axes[2].set_xticks(range(len(df['Activity'].unique())))
        axes[2].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 4. Variabilité des tendances locales
        sns.boxplot(data=df, x='Activity', y=local_std, ax=axes[3])
        axes[3].set_title('Variabilité des tendances locales')
        axes[3].set_xticks(range(len(df['Activity'].unique())))
        axes[3].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        plt.tight_layout()
        filename = f'{prefix}_trend_analysis.png'
        plt.savefig(f'{self.output_dir}/features/{filename}')
        plt.close()

    def create_cyclic_analysis_plots(self, df, feature_names, location, sensor_type=None):
        """Visualise les caractéristiques cycliques"""
        prefix = f"{location}_{sensor_type}" if sensor_type else f"{location}"
        cyclic_features = [f for f in feature_names if f.startswith(prefix) and 
                          any(cf in f for cf in ['autocorr', 'peak'])]
        
        if not cyclic_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Distribution des pics d'autocorrélation
        autocorr_peak = next(f for f in cyclic_features if 'autocorr_peak' in f)
        sns.boxplot(data=df, x='Activity', y=autocorr_peak, ax=axes[0])
        axes[0].set_title('Force de la périodicité par activité')
        axes[0].set_xticks(range(len(df['Activity'].unique())))
        axes[0].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 2. Distribution des lags d'autocorrélation
        autocorr_lag = next(f for f in cyclic_features if 'autocorr_lag' in f)
        sns.boxplot(data=df, x='Activity', y=autocorr_lag, ax=axes[1])
        axes[1].set_title('Période dominante par activité')
        axes[1].set_xticks(range(len(df['Activity'].unique())))
        axes[1].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 3. Nombre de pics
        peak_count = next(f for f in cyclic_features if 'peak_count' in f)
        sns.boxplot(data=df, x='Activity', y=peak_count, ax=axes[2])
        axes[2].set_title('Nombre de pics par activité')
        axes[2].set_xticks(range(len(df['Activity'].unique())))
        axes[2].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 4. Distance moyenne entre pics
        peak_distance = next(f for f in cyclic_features if 'peak_mean_distance' in f)
        sns.boxplot(data=df, x='Activity', y=peak_distance, ax=axes[3])
        axes[3].set_title('Distance moyenne entre pics')
        axes[3].set_xticks(range(len(df['Activity'].unique())))
        axes[3].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        plt.tight_layout()
        filename = f'{prefix}_cyclic_analysis.png'
        plt.savefig(f'{self.output_dir}/features/{filename}')
        plt.close()

    def create_feature_comparison_plot(self, df, feature_names, location, sensor_type=None):
        """Compare les différentes caractéristiques pour identifier les plus discriminantes"""
        prefix = f"{location}_{sensor_type}" if sensor_type else f"{location}"
        all_features = [f for f in feature_names if f.startswith(prefix)]
        
        if not all_features:
            return
        
        print(f"\nAnalyse pour {prefix}:")
        print(f"Nombre de features trouvées: {len(all_features)}")
        
        # Calculer le score de discrimination pour chaque feature
        f_scores = {}
        feature_types = {}  # Dictionnaire pour stocker le type de chaque feature
        
        # Définir les patterns pour chaque type de feature
        statistical_patterns = ['mean', 'std', 'median', 'max', 'min', 'q1', 'q3', 
                                'skew', 'kurtosis', 'var', 'stability', 'valid_ratio']
        temporal_patterns = ['slope', 'trend', 'autocorr', 'peak']
        frequency_patterns = ['freq', 'spectral']
        
        for feature in all_features:
            # Déterminer le type de feature
            if any(pattern in feature.lower() for pattern in statistical_patterns):
                feature_types[feature] = 'statistical'
            elif any(pattern in feature.lower() for pattern in temporal_patterns):
                feature_types[feature] = 'temporal'
            elif any(pattern in feature.lower() for pattern in frequency_patterns):
                feature_types[feature] = 'frequency'
            else:
                feature_types[feature] = 'other'
                
            # Calcul du F-score comme avant
            groups = [group[feature].values for name, group in df.groupby('Activity')]
            clean_groups = []
            for group in groups:
                clean_group = group[~np.isnan(group)]
                if len(clean_group) > 0:
                    clean_groups.append(clean_group)
            
            # Vérifier qu'il reste assez de groupes pour l'analyse
            if len(clean_groups) < 2:
                # print(f"Feature {feature}: Pas assez de groupes valides après nettoyage")
                continue
            
            # Vérifier si toutes les valeurs sont constantes
            if all(np.allclose(g, g[0]) for g in clean_groups):
                # print(f"Feature {feature}: Valeurs constantes")
                continue
                
            try:
                f_stat, p_val = stats.f_oneway(*clean_groups)
                if np.isfinite(f_stat):
                    f_scores[feature] = f_stat
                #     print(f"Feature {feature}: F-stat = {f_stat:.2f}, p-value = {p_val:.4f}")
                # else:
                #     print(f"Feature {feature}: F-stat non finie")
            except Exception as e:
                # print(f"Feature {feature}: Erreur - {str(e)}")
                continue
                
        
        if not f_scores:
            # print(f"Aucun score F valide pour {prefix}")
            return
        
        # Créer un plot des scores avec des couleurs différentes
        plt.figure(figsize=(15, 8))
        features_sorted = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)
        features_to_plot = features_sorted[:20] if len(features_sorted) > 20 else features_sorted
        
        # Définir les couleurs pour chaque type
        color_map = {
            'statistical': '#2ecc71',  # Vert
            'temporal': '#3498db',     # Bleu
            'frequency': '#e74c3c',    # Rouge
            'other': '#95a5a6'         # Gris
        }
        
        # Créer les barres avec les couleurs appropriées
        x = range(len(features_to_plot))
        bars = plt.bar(x, [f[1] for f in features_to_plot])
        
        # Colorer chaque barre selon son type
        for idx, (feature, _) in enumerate(features_to_plot):
            bars[idx].set_color(color_map[feature_types[feature]])
        
        # Ajouter la légende
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=type_name.capitalize())
                          for type_name, color in color_map.items()
                          if any(feature_types[f[0]] == type_name for f in features_to_plot)]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.xticks(x, [f[0] for f in features_to_plot], rotation=45, ha='right')
        plt.title(f'Top Features Discriminantes - {prefix}')
        plt.xlabel('Features')
        plt.ylabel('F-score')
        plt.tight_layout()
        
        filename = f'{prefix}_feature_comparison.png'
        plt.savefig(f'{self.output_dir}/features/{filename}')
        plt.close()

    def create_frequency_analysis_plots(self, df, feature_names, location, sensor_type=None):
        """Visualise les caractéristiques fréquentielles"""
        prefix = f"{location}_{sensor_type}" if sensor_type else f"{location}"
        freq_features = [f for f in feature_names if f.startswith(prefix) and 
                        any(ff in f for ff in ['freq', 'spectral'])]
        
        if not freq_features:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Distribution des fréquences dominantes
        dom_freq = next(f for f in freq_features if 'dominant_freq' in f)
        sns.boxplot(data=df, x='Activity', y=dom_freq, ax=axes[0])
        axes[0].set_title('Fréquence dominante par activité')
        axes[0].set_xticks(range(len(df['Activity'].unique())))
        axes[0].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 2. Amplitude des fréquences dominantes
        freq_amp = next(f for f in freq_features if 'freq_amplitude' in f)
        sns.boxplot(data=df, x='Activity', y=freq_amp, ax=axes[1])
        axes[1].set_title('Amplitude de la fréquence dominante')
        axes[1].set_xticks(range(len(df['Activity'].unique())))
        axes[1].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 3. Centroïde spectral
        centroid = next(f for f in freq_features if 'spectral_centroid' in f)
        sns.boxplot(data=df, x='Activity', y=centroid, ax=axes[2])
        axes[2].set_title('Centre de masse spectral')
        axes[2].set_xticks(range(len(df['Activity'].unique())))
        axes[2].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 4. Dispersion spectrale
        spread = next(f for f in freq_features if 'spectral_spread' in f)
        sns.boxplot(data=df, x='Activity', y=spread, ax=axes[3])
        axes[3].set_title('Dispersion spectrale')
        axes[3].set_xticks(range(len(df['Activity'].unique())))
        axes[3].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 5. Point de coupure spectral
        rolloff = next(f for f in freq_features if 'spectral_rolloff' in f)
        sns.boxplot(data=df, x='Activity', y=rolloff, ax=axes[4])
        axes[4].set_title('Point de coupure spectral')
        axes[4].set_xticks(range(len(df['Activity'].unique())))
        axes[4].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 6. Flux spectral
        flux = next(f for f in freq_features if 'spectral_flux' in f)
        sns.boxplot(data=df, x='Activity', y=flux, ax=axes[5])
        axes[5].set_title('Flux spectral')
        axes[5].set_xticks(range(len(df['Activity'].unique())))
        axes[5].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        plt.suptitle(f'Analyse Fréquentielle - {prefix}', y=1.02, fontsize=16)
        plt.tight_layout()
        
        filename = f'{prefix}_frequency_analysis.png'
        plt.savefig(f'{self.output_dir}/features/{filename}')
        plt.close()

    def create_shape_analysis_plots(self, df, feature_names, location, sensor_type=None):
        """Visualise les caractéristiques de forme"""
        prefix = f"{location}_{sensor_type}" if sensor_type else f"{location}"
        shape_features = [f for f in feature_names if f.startswith(prefix) and 
                         any(sf in f for sf in ['peak_count', 'peak_width_mean', 'zero_crossings'])]
        
        if not shape_features:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.flatten()
        
        # 1. Nombre de pics
        peak_count = next(f for f in shape_features if 'peak_count' in f)
        sns.boxplot(data=df, x='Activity', y=peak_count, ax=axes[0])
        axes[0].set_title('Nombre de pics par activité')
        axes[0].set_xticks(range(len(df['Activity'].unique())))
        axes[0].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 2. Largeur moyenne des pics
        peak_width_mean = next(f for f in shape_features if 'peak_width_mean' in f)
        sns.boxplot(data=df, x='Activity', y=peak_width_mean, ax=axes[1])
        axes[0].set_title('Largeur moyenne des pics par activité')
        axes[0].set_xticks(range(len(df['Activity'].unique())))
        axes[0].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        # 3. Passages par zéro
        zero_crossings = next(f for f in shape_features if 'zero_crossings' in f)
        sns.boxplot(data=df, x='Activity', y=zero_crossings, ax=axes[2])
        axes[1].set_title('Passages par zéro par activité')
        axes[1].set_xticks(range(len(df['Activity'].unique())))
        axes[1].set_xticklabels(df['Activity'].unique(), rotation=45, ha='right')
        
        plt.suptitle(f'Analyse des Caractéristiques de Forme - {prefix}', y=1.02, fontsize=16)
        plt.tight_layout()
        
        filename = f'{prefix}_shape_analysis.png'
        plt.savefig(f'{self.output_dir}/features/{filename}')
        plt.close()