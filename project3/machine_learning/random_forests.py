import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_transformed_data

from toy_script import write_submission

class ModelAnalyzer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def plot_learning_curves(self, estimator):
        """Tracer les courbes d'apprentissage avec des intervalles de confiance"""
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, self.X_train, self.y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(12, 8))
        plt.plot(train_sizes, train_mean, label='Training score', color='#2ecc71', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2ecc71')
        
        plt.plot(train_sizes, val_mean, label='Cross-validation score', color='#e74c3c', marker='o')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#e74c3c')
        
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Learning Curves\nModel Performance vs Training Size', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ajouter des annotations
        plt.annotate(f'Final CV Score: {val_mean[-1]:.3f} ± {val_std[-1]:.3f}',
                    xy=(train_sizes[-1], val_mean[-1]),
                    xytext=(train_sizes[-1]*0.8, val_mean[-1]),
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, model, feature_names):
        """Visualiser l'importance des features avec des graphiques détaillés"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Créer une figure avec plusieurs sous-graphiques
            fig = plt.figure(figsize=(20, 15))
            gs = plt.GridSpec(2, 2, figure=fig)
            
            # 1. Top 20 Features (Graphique principal)
            ax1 = fig.add_subplot(gs[0, :])
            n_features = 20
            top_indices = indices[:n_features]
            
            feature_importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': importances[top_indices]
            })
            
            # Créer le barplot avec un dégradé de couleurs basé sur l'importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
            bars = ax1.barh(range(len(feature_importance_df)), 
                           feature_importance_df['Importance'],
                           color=colors)
            
            # Ajouter les labels et pourcentages
            for i, (v, feature) in enumerate(zip(feature_importance_df['Importance'], 
                                               feature_importance_df['Feature'])):
                ax1.text(v, i, f' {v*100:.1f}%', va='center', fontsize=10)
                ax1.text(-0.01, i, feature, ha='right', va='center', fontsize=10)
            
            ax1.set_title('Top 20 Most Important Features', fontsize=14, pad=20)
            ax1.set_xlabel('Importance Score', fontsize=12)
            ax1.set_yticks([])
            ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # 2. Distribution des importances (en bas à gauche)
            ax2 = fig.add_subplot(gs[1, 0])
            sns.histplot(importances[importances > 0.001], bins=30, ax=ax2)
            ax2.set_title('Distribution of Feature Importances\n(>0.1%)', fontsize=12)
            ax2.set_xlabel('Importance Score')
            ax2.set_ylabel('Count')
            
            # 3. Importance cumulée (en bas à droite)
            ax3 = fig.add_subplot(gs[1, 1])
            cumsum = np.cumsum(importances[indices])
            n_features_90 = np.where(cumsum >= 0.9)[0][0] + 1
            
            ax3.plot(range(1, len(cumsum) + 1), cumsum, 'b-')
            ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
            ax3.axvline(x=n_features_90, color='r', linestyle='--', alpha=0.5)
            ax3.fill_between(range(1, len(cumsum) + 1), cumsum, alpha=0.3)
            ax3.set_title('Cumulative Importance', fontsize=12)
            ax3.set_xlabel('Number of Features')
            ax3.set_ylabel('Cumulative Importance')
            ax3.annotate(f'90% importance:\n{n_features_90} features', 
                        xy=(n_features_90, 0.9),
                        xytext=(n_features_90+10, 0.85),
                        arrowprops=dict(facecolor='black', shrink=0.05))
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Sauvegarder un résumé détaillé
            with open('feature_importance_summary.txt', 'w') as f:
                f.write("Feature Importance Analysis\n")
                f.write("="*50 + "\n\n")
                f.write(f"Number of features needed for 90% importance: {n_features_90}\n\n")
                f.write("Top 20 Most Important Features:\n")
                for idx, row in feature_importance_df.iterrows():
                    f.write(f"{idx+1}. {row['Feature']}: {row['Importance']*100:.2f}%\n")
                
                # Ajouter des statistiques supplémentaires
                f.write("\nFeature Importance Statistics:\n")
                f.write(f"Mean importance: {importances.mean()*100:.2f}%\n")
                f.write(f"Median importance: {np.median(importances)*100:.2f}%\n")
                f.write(f"Standard deviation: {importances.std()*100:.2f}%\n")
                f.write(f"Number of features with >1% importance: {sum(importances > 0.01)}\n")

    def plot_grid_search_results(self, grid_search):
        """Visualiser les résultats de la recherche d'hyperparamètres avec plus de détails"""
        results = pd.DataFrame(grid_search.cv_results_)
        
        plt.figure(figsize=(15, 10))
        
        # Créer plusieurs sous-graphiques
        gs = plt.GridSpec(2, 2)
        
        # 1. Heatmap des scores moyens
        ax1 = plt.subplot(gs[0, :])
        pivot_table = results.pivot_table(
            values='mean_test_score',
            index='param_classifier__min_samples_split',
            columns='param_classifier__n_estimators'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Grid Search Scores: min_samples_split vs n_estimators', pad=20)
        
        # 2. Distribution des scores
        ax2 = plt.subplot(gs[1, 0])
        sns.violinplot(data=results, y='mean_test_score', ax=ax2)
        ax2.set_title('Distribution of CV Scores')
        ax2.set_ylabel('Score')
        
        # 3. Tous les configurations triées
        ax3 = plt.subplot(gs[1, 1])
        n_configs = len(results)  # Utiliser le nombre réel de configurations
        top_n = results.nlargest(n_configs, 'mean_test_score')
        sns.barplot(data=top_n, x=range(n_configs), y='mean_test_score', ax=ax3)
        ax3.set_title('All Configurations Ranked')
        ax3.set_xlabel('Configuration Rank')
        ax3.set_ylabel('Score')
        
        # Rotation des labels pour une meilleure lisibilité
        ax3.set_xticklabels(range(1, n_configs + 1), rotation=45)
        
        plt.tight_layout()
        plt.savefig('grid_search_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarder un résumé des meilleurs résultats
        with open('grid_search_summary.txt', 'w') as f:
            f.write("Grid Search Results Summary\n")
            f.write("="*50 + "\n\n")
            f.write("Best Configuration:\n")
            f.write(f"Score: {grid_search.best_score_:.4f}\n")
            for param, value in grid_search.best_params_.items():
                f.write(f"{param}: {value}\n")
            
            f.write("\nAll Configurations Ranked:\n")
            for idx, row in top_n.iterrows():
                f.write(f"\nRank {idx+1}:\n")
                f.write(f"Score: {row['mean_test_score']:.4f}\n")
                for param in grid_search.param_grid.keys():
                    f.write(f"{param}: {row[f'param_{param}']}\n")

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """Visualiser la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()


def main():
    # Charger les données
    X_train, y_train, X_test, feature_names = load_transformed_data()
    
    # Diviser les données en ensembles d'apprentissage et d'évaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Créer l'analyseur de modèle
    analyzer = ModelAnalyzer(X_train, y_train)
    
    # Définir le pipeline et la grille de paramètres
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])


    ##########################################################
    # * Paramètres à optimiser
    ##########################################################

    # param_grid = {
    #     'classifier__n_estimators': [100, 200, 300, 400, 500],
    #     'classifier__max_depth': [10, 20, 30, 40, 50, None],
    #     'classifier__min_samples_split': [2, 5, 10, 15],
    #     'classifier__min_samples_leaf': [1, 2, 4],
    #     'classifier__max_features': ['sqrt', 'log2', None],
    #     'classifier__max_leaf_nodes': [None, 50, 100, 200],
    #     'classifier__min_impurity_decrease': [0.0, 0.1, 0.2]
    # }

    # Paramètres à optimiser
    param_grid = {
        'classifier__n_estimators': [300],
        'classifier__max_depth': [20],
        'classifier__min_samples_split': [2, 5, 10],
    }
    
    # * Recherche d'hyperparamètres / Entraînement

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, 
                             scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # * Visualisations

    print("Generating visualizations...")
    analyzer.plot_learning_curves(grid_search.best_estimator_)
    analyzer.plot_grid_search_results(grid_search)
    analyzer.plot_feature_importance(
        grid_search.best_estimator_.named_steps['classifier'],
        feature_names  
    )

    print("Best parameters found:", grid_search.best_params_)

    # Évaluer le modèle sur l'ensemble de validation
    val_predictions = grid_search.predict(X_val)
    val_score = f1_score(y_val, val_predictions, average='weighted')
    print(f"Validation F1 Score: {val_score:.3f}")

    # Utiliser le meilleur modèle pour les prédictions sur X_test
    predictions = grid_search.predict(X_test)
    
    # Sauvegarder le modèle et les prédictions
    # joblib.dump(grid_search.best_estimator_, 'random_forest_model.pkl')
    # write_submission(predictions, 'submissions/random_forest_submission.csv')

if __name__ == "__main__":
    main()
