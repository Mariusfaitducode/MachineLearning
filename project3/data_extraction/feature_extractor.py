import numpy as np
from scipy import stats

class FeatureExtractor:
    """Classe gérant l'extraction des caractéristiques"""
    
    def __init__(self):
        self.feature_names = [
            'mean', 'std', 'median', 'max', 'min', 'q1', 'q3',
            'skew', 'kurtosis', 'total_var', 'mean_var',
            'stability', 'valid_ratio'
        ]
    
    def extract_statistical_features(self, data):
        """Extrait les caractéristiques statistiques de base"""
        if np.all(data == -999999.99):
            return [np.nan] * 13
        
        data = np.where(data == -999999.99, np.nan, data)
        
        features = []
        # Caractéristiques statistiques de base
        features.extend([
            np.nanmean(data),           # Moyenne
            np.nanstd(data),            # Écart-type
            np.nanmedian(data),         # Médiane
            np.nanmax(data),            # Maximum
            np.nanmin(data),            # Minimum
            np.nanpercentile(data, 25), # 1er quartile
            np.nanpercentile(data, 75)  # 3ème quartile
        ])
        
        # Calcul de l'asymétrie et du kurtosis
        try:
            if np.nanstd(data) < 1e-10:
                features.extend([0.0, 0.0])
            else:
                features.extend([
                    stats.skew(data, nan_policy='omit'),
                    stats.kurtosis(data, nan_policy='omit')
                ])
        except:
            features.extend([np.nan, np.nan])
        
        # Caractéristiques de forme
        try:
            diff_data = np.diff(data[~np.isnan(data)])
            if len(diff_data) > 0:
                features.extend([
                    np.sum(np.abs(diff_data)),     # Variation totale
                    np.mean(np.abs(diff_data)),    # Variation moyenne
                    np.std(diff_data)              # Stabilité
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        except:
            features.extend([np.nan, np.nan, np.nan])
        
        # Proportion de données valides
        features.append(len(data[~np.isnan(data)]) / len(data))
        
        return features

    def get_feature_names(self, location, sensor_type=None, axis=None):
        """Génère les noms des caractéristiques selon le capteur"""
        base = f"{location}"
        if sensor_type:
            base += f"_{sensor_type}"
        if axis:
            base += f"_{axis}"
        
        return [f"{base}_{feat}" for feat in self.feature_names]
    

    def extract_temporal_features(self, data):
        """Extrait les caractéristiques temporelles"""
        features = []
        # Tendance
        self._extract_trend_features(data, features)
        # Cycles
        self._extract_cyclic_features(data, features)
        # Forme
        self._extract_shape_features(data, features)
        return features
    
    def extract_frequency_features(self, data):
        """Extrait les caractéristiques fréquentielles"""
        features = []
        # Analyse spectrale
        self._extract_spectral_features(data, features)
        # Énergie
        self._extract_energy_features(data, features)
        return features
    
    def _extract_trend_features(self, data, features):
        """Extrait les caractéristiques de tendance"""
        pass
    
    def _extract_cyclic_features(self, data, features):
        """Extrait les caractéristiques cycliques"""
        pass
    
    def _extract_shape_features(self, data, features):
        """Extrait les caractéristiques de forme"""
        pass
    
    def _extract_spectral_features(self, data, features):
        """Extrait les caractéristiques spectrales"""
        pass
    
    def _extract_energy_features(self, data, features):
        """Extrait les caractéristiques énergétiques"""
        pass