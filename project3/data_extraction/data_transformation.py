import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_script import load_data
from feature_extractor import FeatureExtractor
from preprocessing import Preprocessor
from visualizer import Visualizer
from timing_decorator import timing_decorator
from tqdm import tqdm


class DataTransformer:
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = Preprocessor()
        self.visualizer = Visualizer()
        
        self.activity_names = {
            1: 'Lying', 2: 'Sitting', 3: 'Standing', 4: 'Walking very slow',
            5: 'Normal walking', 6: 'Nordic walking', 7: 'Running', 
            8: 'Ascending stairs', 9: 'Descending stairs', 10: 'Cycling',
            11: 'Ironing', 12: 'Vacuum cleaning', 13: 'Rope jumping', 
            14: 'Playing soccer'
        }
        
        # Définir les groupes de capteurs
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

    # @timing_decorator
    def transform_data(self):
        print("Starting data transformation...")
        transformed_data = []
        feature_names = []
        
        total_samples = len(self.X_train)
        
        # Utiliser tqdm pour la barre de progression
        for i in tqdm(range(total_samples), desc="Transforming data"):
            sample = self.X_train[i]
            sample_features = []
            for location, sensors in self.sensor_groups.items():
                self._process_sensor_group(sample, location, sensors, 
                                        sample_features, feature_names)
            transformed_data.append(sample_features)
        
        print("Data transformation completed")
        return np.array(transformed_data), feature_names
    
    # @timing_decorator
    def _process_sensor_group(self, sample, location, sensors, 
                             sample_features, feature_names):
        """
        Traite un groupe de capteurs de manière unifiée
        
        Args:
            sample: Données brutes d'un échantillon
            location: Emplacement du capteur (Heart, Hand, Chest, Foot)
            sensors: Configuration des capteurs (liste ou dictionnaire)
            sample_features: Liste des caractéristiques extraites
            feature_names: Liste des noms des caractéristiques
        """
        if isinstance(sensors, list):
            # Cas du capteur cardiaque (Heart)
            sensor_configs = [('', sensors[0], None)]
        else:
            # Cas des autres capteurs (Hand, Chest, Foot)
            sensor_configs = []
            for sensor_type, sensor_ids in sensors.items():
                if sensor_type == 'Temperature':
                    sensor_configs.append((sensor_type, sensor_ids[0], None))
                else:
                    # Capteurs avec axes (Acceleration, Gyroscope, Magnetometer)
                    sensor_configs.extend([
                        (sensor_type, sensor_id, axis)
                        for sensor_id, axis in zip(sensor_ids, ['x', 'y', 'z'])
                    ])
        
        # Traitement unifié pour tous les types de capteurs
        for sensor_type, sensor_id, axis in sensor_configs:
            # Extraction des données du capteurq
            start = (sensor_id - 2) * 512
            end = start + 512
            raw_data = sample[start:end]
            
            # ! Prétraitement des données
            # cleaned_data = self.preprocessor.process(raw_data, sensor_type)

            # print(f"Raw data shape: {raw_data.shape}")
            # print('extract features code now')            
            # Extraction des caractéristiques sur les données nettoyées
            features = self.feature_extractor.extract_features(raw_data)
            sample_features.extend(features)
            
            # Ajout des noms de caractéristiques si nécessaire
            if len(feature_names) < len(sample_features):
                feature_names.extend(self.feature_extractor.get_feature_names(
                    location, sensor_type, axis
                ))

    def save_transformed_data_csv(self, transformed_data, feature_names, filename='transformed_data.csv'):
        """Sauvegarde les données transformées au format CSV"""
        df = pd.DataFrame(transformed_data, columns=feature_names)
        df.to_csv(filename, index=False)
        print(f"Transformed data saved to {filename}")

def main():
    # Load full dataset
    X_train, y_train, X_test, subject_ids_train, subject_ids_test = load_data(max_size=100)

    transformer = DataTransformer(X_train, y_train, X_test)
    transformed_data, feature_names = transformer.transform_data()
    transformer.visualizer.visualize_features(transformed_data, feature_names, [transformer.activity_names[y] for y in y_train])

    transformer.save_transformed_data_csv(transformed_data, feature_names)

if __name__ == "__main__":
    main()
