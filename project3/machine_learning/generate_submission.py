import numpy as np
import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toy_script import write_submission
from load_data import load_transformed_data
# from data_extraction.data_transformation import DataTransformer

def generate_submission():
    print("Loading test data...")
    # Charger les données de test
    X_train, y_train, X_test = load_transformed_data()
    
    # print("Transforming test data...")
    # # Transformer les données de test
    # transformer = DataTransformer(X_train, y_train, X_test)
    # transformed_test_data, _ = transformer.transform_data()
    
    print("Loading trained model...")
    # Charger le modèle entraîné
    model = joblib.load('best_random_forest_model.pkl')
    
    print("Making predictions...")
    # Faire les prédictions
    predictions = model.predict(X_test)
    
    print("Generating submission file...")
    # Créer le fichier de soumission
    submission_path = 'submissions/random_forest_submission.csv'
    write_submission(predictions, submission_path)
    
    print("Done! Submission file created at:", submission_path)
    
    # Afficher quelques statistiques sur les prédictions
    print("\nPrediction statistics:")
    print("Number of predictions:", len(predictions))
    print("Unique classes predicted:", np.unique(predictions))
    value_counts = pd.Series(predictions).value_counts().sort_index()
    print("\nClass distribution:")
    for class_id, count in value_counts.items():
        print(f"Class {int(class_id)}: {count} samples")

if __name__ == "__main__":
    # Créer le dossier submissions s'il n'existe pas
    os.makedirs('submissions', exist_ok=True)
    generate_submission()