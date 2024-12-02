import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        self.visualize_correlation_matrices(df, feature_names)


    def visualize_time_series(self, data, sensor_info):
        """Visualise les séries temporelles"""
        pass
    
    def visualize_frequency_domain(self, data, sensor_info):
        """Visualise les caractéristiques fréquentielles"""
        pass