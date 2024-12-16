from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def compare_models(X_train, y_train, X_val, y_val):
    # Encoder les labels pour commencer à 0
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    # Définir les modèles à comparer
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                random_state=42
            ))
        ]),
        
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ))
        ]),
        
        # 'SVM': Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', SVC(
        #         kernel='rbf',
        #         C=1.0,
        #         probability=True,
        #         random_state=42
        #     ))
        # ])
    }
    
    # Résultats pour chaque modèle
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Cross-validation sur données d'entraînement
        cv_scores = cross_val_score(
            model, X_train, y_train_encoded, 
            cv=5, 
            scoring='f1_weighted'
        )
        
        # Entraînement sur tout le jeu d'entraînement
        model.fit(X_train, y_train_encoded)
        
        # Score sur validation
        val_score = model.score(X_val, y_val_encoded)
        
        results.append({
            'Model': name,
            'CV Mean Score': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'Validation Score': val_score,
        })
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    return results_df

# Utilisation
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from load_data import load_transformed_data
    from sklearn.model_selection import train_test_split

    # Charger les données
    X_train_full, y_train_full, X_test, feature_names = load_transformed_data()

    # Diviser en train et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )

    results = compare_models(X_train, y_train, X_val, y_val)
    print("\nModel Comparison Results:")
    print(results.to_string(index=False))
