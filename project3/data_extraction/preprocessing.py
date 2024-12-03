import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

class Preprocessor:
    """Classe gérant le prétraitement des données brutes"""
    
    def __init__(self):
        pass
    
    def process(self, data, sensor_type):
        """Prétraitement adapté au type de capteur"""
        if np.all(data == -999999.99):
            return data
        
        # Application du filtrage selon le type de capteur
        if sensor_type in ['Acceleration', 'Gyroscope']:
            # Filtrage plus agressif pour les capteurs de mouvement
            data = self._filter_noise(data, kernel_size=3)
        elif sensor_type in ['Magnetometer']:
            # Filtrage léger pour le magnétomètre
            data = self._filter_noise(data, kernel_size=3)
        elif sensor_type in ['Heart']:
            # Filtrage médian spécifique pour la fréquence cardiaque
            data = self._filter_outliers(data)
        
        # Normalisation uniforme pour tous les capteurs
        return self._normalize(data)
    
    def _filter_noise(self, data, kernel_size):
        """Filtre le bruit avec un filtre médian"""
        return medfilt(data, kernel_size=kernel_size)
    
    def _normalize(self, data):
        """Normalisation standard (z-score) pour tous les capteurs"""
        if np.std(data) < 1e-10:  # Éviter la division par zéro
            return np.zeros_like(data)
        return (data - np.mean(data)) / np.std(data)
    
    def _filter_outliers(self, data):
        """Filtre les valeurs aberrantes pour la fréquence cardiaque"""
        # Utilise la médiane et l'écart interquartile
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return np.clip(data, lower_bound, upper_bound)