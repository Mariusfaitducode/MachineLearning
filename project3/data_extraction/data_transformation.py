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

    def transform_data(self):
        """Transforme les données en extrayant les caractéristiques"""
        feature_names = []
        transformed_data = []
        
        for sample in self.X_train:
            sample_features = []
            for location, sensors in self.sensor_groups.items():
                self._process_sensor_group(sample, location, sensors, 
                                        sample_features, feature_names)
            transformed_data.append(sample_features)
        
        return np.array(transformed_data), feature_names
    
    def _process_sensor_group(self, sample, location, sensors, 
                            sample_features, feature_names):
        """Traite un groupe de capteurs"""
        if isinstance(sensors, list):
            # Cas du capteur cardiaque (Heart)
            for sensor_id in sensors:
                start = (sensor_id - 2) * 512
                end = start + 512
                features = self.feature_extractor.extract_statistical_features(
                    sample[start:end]
                )
                sample_features.extend(features)
                if len(feature_names) < len(sample_features):
                    feature_names.extend(self.feature_extractor.get_feature_names(
                        location
                    ))
        else:
            # Cas des autres capteurs (Hand, Chest, Foot)
            for sensor_type, sensor_ids in sensors.items():
                if sensor_type == 'Temperature':
                    # Cas de la température (une seule valeur)
                    start = (sensor_ids[0] - 2) * 512
                    end = start + 512
                    features = self.feature_extractor.extract_statistical_features(
                        sample[start:end]
                    )
                    sample_features.extend(features)
                    if len(feature_names) < len(sample_features):
                        feature_names.extend(self.feature_extractor.get_feature_names(
                            location, sensor_type
                        ))
                else:
                    # Cas des capteurs avec axes (Acceleration, Gyroscope, Magnetometer)
                    axes = ['x', 'y', 'z']
                    for axis, sensor_id in zip(axes, sensor_ids):
                        start = (sensor_id - 2) * 512
                        end = start + 512
                        features = self.feature_extractor.extract_statistical_features(
                            sample[start:end]
                        )
                        sample_features.extend(features)
                        if len(feature_names) < len(sample_features):
                            feature_names.extend(self.feature_extractor.get_feature_names(
                                location, sensor_type, axis
                            ))

def main():
    # Load full dataset
    X_train, y_train, X_test, subject_ids_train, subject_ids_test = load_data(max_size=None)

    transformer = DataTransformer(X_train, y_train, X_test)
    transformed_data, feature_names = transformer.transform_data()
    transformer.visualizer.visualize_features(transformed_data, feature_names, [transformer.activity_names[y] for y in y_train])

if __name__ == "__main__":
    main()
