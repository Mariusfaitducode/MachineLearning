import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from functools import lru_cache

from timing_decorator import timing_decorator



class FeatureExtractor:
    """Classe gérant l'extraction des caractéristiques"""
    
    def __init__(self):
        self.statistical_features = [
            'mean', 'std', 'median', 'max', 'min', 'q1', 'q3',
            'skew', 'kurtosis', 'total_var', 'mean_var',
            'stability', 'valid_ratio'
        ]
        
        self.trend_features = [
            'slope', 'slope_error', 'local_trend_mean', 'local_trend_std'
        ]
        
        self.cyclic_features = [
            'autocorr_peak', 'autocorr_lag', 'peak_count', 'peak_mean_distance'
        ]
        
        self.frequency_features = [
            'dominant_freq', 'freq_amplitude', 
            'spectral_centroid', 'spectral_spread',
            'spectral_rolloff', 'spectral_flux'
        ]
        
        self.shape_features = [
            'peak_width_mean', 'zero_crossings'
        ]
        
        self.feature_names = (
            self.statistical_features + 
            self.trend_features + 
            self.cyclic_features +
            self.frequency_features +
            self.shape_features
        )
    
    # @timing_decorator
    def extract_features(self, data):
        """Extrait toutes les caractéristiques"""
        if np.all(data == -999999.99):
            return [np.nan] * len(self.feature_names)
            
        # Nettoyage basique des données
        data = np.where(data == -999999.99, np.nan, data)
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) < 2:  # Pas assez de données valides
            return [np.nan] * len(self.feature_names)
            
        features = []
        # Features statistiques existantes
        features.extend(self.extract_statistical_features(data))
        # Nouvelles features
        features.extend(self._extract_trend_features(valid_data))
        features.extend(self._extract_cyclic_features(valid_data))
        features.extend(self._extract_frequency_features(valid_data))
        features.extend(self._extract_shape_features(valid_data))
        
        return features
    
    def extract_statistical_features(self, data):
        """Extrait les caractéristiques statistiques de base"""
        if np.all(data == np.nan) or np.all(data == -999999.99):
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
    
    # @timing_decorator
    def _extract_trend_features(self, data):
        """Extrait les caractéristiques de tendance
        
        Cette méthode analyse les tendances dans les données en utilisant une approche de fenêtre glissante:
        
        1. Découpe les données en fenêtres de taille window_size (min entre 50 et len(data)/4)
        2. Pour chaque fenêtre, calcule la pente de la régression linéaire
        3. Extrait 4 caractéristiques à partir des pentes:
           - Pente globale: pente de la première fenêtre
           - Variation des pentes: écart-type des pentes de toutes les fenêtres  
           - Moyenne des pentes: moyenne des pentes de toutes les fenêtres
           - IQR des pentes: écart interquartile (Q3-Q1) des pentes
           
        Args:
            data: Série temporelle à analyser
            
        Returns:
            Liste de 4 caractéristiques de tendance, ou [nan, nan, nan, nan] en cas d'erreur
        """
        try:
            # Calcul vectorisé des tendances locales
            window_size = min(50, len(data) // 4)
            windows = np.lib.stride_tricks.sliding_window_view(data, window_size)
            x = np.arange(window_size)
            
            # Calcul vectorisé des pentes
            slopes = np.polyfit(x, windows.T, 1)[0]
            
            return [
                slopes[0],              # Pente globale
                np.std(slopes),         # Variation des pentes
                np.mean(slopes),        # Moyenne des pentes
                np.percentile(slopes, 75) - np.percentile(slopes, 25)  # IQR des pentes
            ]
        except:
            return [np.nan] * 4
    
    # @timing_decorator
    def _extract_cyclic_features(self, data):
        """Extrait les caractéristiques cycliques
        
        Cette méthode analyse les aspects cycliques/périodiques du signal en calculant:
        
        1. L'autocorrélation du signal:
           - Normalise d'abord le signal en soustrayant la moyenne
           - Calcule l'autocorrélation via np.correlate
           - Normalise par la variance pour avoir des valeurs entre -1 et 1
           
        2. Détection des pics dans l'autocorrélation:
           - Cherche les pics significatifs (hauteur > 0.2) espacés d'au moins 10 points
           - Extrait la position et la valeur du premier pic trouvé
           - Un pic indique une périodicité dans le signal
           
        3. Analyse des pics dans le signal original:
           - Détecte les pics espacés d'au moins 5 points
           - Compte le nombre total de pics
           - Calcule la distance moyenne entre pics successifs
           
        Args:
            data: Série temporelle à analyser
            
        Returns:
            Liste de 4 caractéristiques cycliques:
            - Valeur du premier pic d'autocorrélation 
            - Position (lag) du premier pic
            - Nombre total de pics dans le signal
            - Distance moyenne entre pics successifs
            
            Retourne [nan, nan, nan, nan] en cas d'erreur
        """
        try:
            # Calcul de l'autocorrélation
            n = len(data)
            mean = np.mean(data)
            var = np.var(data)
            
            # Éviter la division par zéro
            if var < 1e-10:  # Si variance quasi-nulle
                return [0, 0, 0, 0]  # Signal constant
            
            normalized_data = data - mean
            acorr = np.correlate(normalized_data, normalized_data, 'full')[n-1:] / (var * n)
            
            # Trouver le premier pic significatif (après lag 0)
            peaks, _ = find_peaks(acorr, height=0.2, distance=10)  # Seuil à 0.1
            if len(peaks) > 0:
                first_peak = peaks[0]
                peak_value = acorr[first_peak]
            else:
                first_peak = 0
                peak_value = 0
                
            # Analyse des pics dans le signal original
            signal_peaks, _ = find_peaks(data, distance=5)  # Distance minimale entre pics
            if len(signal_peaks) > 1:
                peak_distances = np.diff(signal_peaks)
            else:
                peak_distances = [0]
            
            return [
                peak_value,                    # Valeur du pic d'autocorrélation
                first_peak,                    # Lag du premier pic
                len(signal_peaks),             # Nombre de pics
                np.mean(peak_distances)        # Distance moyenne entre pics
            ]
        except:
            return [np.nan] * 4
    
    @lru_cache(maxsize=1000)
    def _compute_autocorrelation(self, data_tuple):
        # Convertir le tuple en array pour les calculs
        data = np.array(data_tuple)
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)
        
        if var < 1e-10:
            return np.zeros(n)
            
        normalized_data = data - mean
        return np.correlate(normalized_data, normalized_data, 'full')[n-1:] / (var * n)
    
    def _extract_frequency_features(self, data):
        """Extrait les caractéristiques fréquentielles
        
        Cette méthode calcule plusieurs caractéristiques dans le domaine fréquentiel:
        
        1. Transformée de Fourier rapide (FFT):
           - Convertit le signal temporel en composantes fréquentielles
           - Ne garde que les fréquences positives pour éviter la redondance
        
        2. Fréquence dominante:
           - Trouve la fréquence ayant la plus grande amplitude
           - Retourne sa valeur et son amplitude
        
        3. Centroïde spectral:
           - Représente le "centre de masse" du spectre fréquentiel
           - Indique où se concentre l'énergie du signal
        
        4. Dispersion spectrale:
           - Mesure l'étalement des fréquences autour du centroïde
           - Une grande dispersion indique un signal riche en fréquences
        
        5. Point de coupure spectral:
           - Fréquence en dessous de laquelle se trouve 85% de l'énergie
           - Caractérise la distribution de l'énergie spectrale
        
        6. Flux spectral:
           - Mesure les variations d'amplitude entre fréquences adjacentes
           - Indique la "rugosité" du spectre
        
        Args:
            data: Série temporelle à analyser
            
        Returns:
            Liste des 6 caractéristiques fréquentielles
        """
        try:
            # Calculer la FFT
            n = len(data)
            fft = np.fft.fft(data)
            freq = np.fft.fftfreq(n)
            
            # Ne garder que les fréquences positives
            pos_mask = freq > 0
            freq = freq[pos_mask]
            fft = np.abs(fft[pos_mask])
            
            # Fréquence dominante
            dominant_idx = np.argmax(fft)
            dominant_freq = freq[dominant_idx]
            freq_amplitude = fft[dominant_idx]
            
            # Centroïde spectral (centre de masse du spectre)
            spectral_centroid = np.sum(freq * fft) / np.sum(fft)
            
            # Dispersion spectrale
            spectral_spread = np.sqrt(np.sum(((freq - spectral_centroid) ** 2) * fft) / np.sum(fft))
            
            # Point de coupure spectral (fréquence en dessous de laquelle se trouve 85% de l'énergie)
            cumsum = np.cumsum(fft)
            rolloff_point = np.where(cumsum >= 0.85 * cumsum[-1])[0][0]
            spectral_rolloff = freq[rolloff_point]
            
            # Flux spectral (taux de changement du spectre)
            spectral_flux = np.mean(np.diff(fft) ** 2)
            
            return [
                dominant_freq,      # Fréquence principale
                freq_amplitude,     # Amplitude de la fréquence principale
                spectral_centroid, # Centre de masse du spectre
                spectral_spread,   # Dispersion des fréquences
                spectral_rolloff,  # Point de coupure spectral
                spectral_flux      # Variation du spectre
            ]
            
        except:
            return [np.nan] * 6
    
    def _extract_shape_features(self, data):
        """Extrait les caractéristiques de forme"""
        try:
            # Détection des pics
            peaks, properties = find_peaks(data, width=1)
            peak_count = len(peaks)
            peak_width_mean = np.mean(properties['widths']) if peak_count > 0 else 0
            
            # Détection des passages par zéro
            zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
            
            features = [        # Nombre de pics
             peak_width_mean,   # Largeur moyenne des pics
             zero_crossings     # Nombre de passages par zéro
            ]
            # print("Shape features:", features)
            return features
        except:
            return [np.nan, np.nan]