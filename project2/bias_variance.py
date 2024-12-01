import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from typing import Callable

def estimate_bias_variance(
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    n_samples: int = 250,
    n_iterations: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Estime le biais, la variance et l'erreur attendue d'un modèle.
    
    Args:
        X: Features du dataset complet
        y: Labels du dataset complet
        estimator: Modèle à évaluer (non entraîné)
        n_samples: Taille de l'échantillon d'apprentissage
        n_iterations: Nombre d'itérations pour l'estimation
        test_size: Proportion du jeu de test
        random_state: Pour la reproductibilité
    
    Returns:
        (expected_error, variance, bias_residual)
    """
    # Séparation initiale en train/test
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Prédictions de tous les modèles sur l'ensemble de test
    predictions = np.zeros((n_iterations, len(y_test)))
    
    np.random.seed(random_state)
    
    # Pour chaque itération
    for i in range(n_iterations):
        # Échantillonnage aléatoire de n_samples exemples
        indices = np.random.choice(len(X_pool), size=n_samples, replace=False)
        X_train = X_pool[indices]
        y_train = y_pool[indices]
        
        # Entraînement du modèle
        model = clone(estimator)
        model.fit(X_train, y_train)
        
        # Prédictions sur l'ensemble de test
        predictions[i] = model.predict(X_test)
    
    # Calcul des métriques
    expected_predictions = np.mean(predictions, axis=0)
    
    # Variance (moyenne des variances pour chaque point)
    variance = np.mean(np.var(predictions, axis=0))
    
    # Erreur attendue (MSE moyen)
    expected_error = np.mean((predictions - y_test.reshape(1, -1)) ** 2)
    
    # Biais² + erreur résiduelle
    bias_residual = expected_error - variance
    
    return expected_error, variance, bias_residual

def evaluate_model_complexity(
    X: np.ndarray,
    y: np.ndarray,
    create_model: Callable,
    param_range: list,
    param_name: str,
    n_samples: int = 1000,
    n_iterations: int = 100
) -> tuple:
    """
    Évalue l'impact d'un hyperparamètre sur le biais et la variance.
    
    Args:
        X: Features
        y: Labels
        create_model: Fonction qui crée le modèle avec un paramètre donné
        param_range: Liste des valeurs du paramètre à tester
        param_name: Nom du paramètre (pour l'affichage)
        n_samples: Taille de l'échantillon d'apprentissage
        n_iterations: Nombre d'itérations pour l'estimation
    
    Returns:
        (errors, variances, bias_residuals)
    """
    errors = []
    variances = []
    bias_residuals = []
    
    for param_value in param_range:
        # Création du modèle avec le paramètre spécifié
        model = create_model(param_value)
        
        # Estimation du biais et de la variance
        error, variance, bias_res = estimate_bias_variance(
            X, y, model, n_samples, n_iterations
        )
        
        errors.append(error)
        variances.append(variance)
        bias_residuals.append(bias_res)
        
        print(f"{param_name}={param_value}: "
              f"Error={error:.4f}, "
              f"Variance={variance:.4f}, "
              f"Bias²+Residual={bias_res:.4f}")
    
    return np.array(errors), np.array(variances), np.array(bias_residuals)

# Fonction utilitaire pour visualiser les résultats
def plot_bias_variance_trade_off(
    param_range: list,
    errors: np.ndarray,
    variances: np.ndarray,
    bias_residuals: np.ndarray,
    param_name: str
):
    """
    Visualise le compromis biais-variance en fonction d'un paramètre.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, errors, 'r-', label='Expected Error')
    plt.plot(param_range, variances, 'b-', label='Variance')
    plt.plot(param_range, bias_residuals, 'g-', label='Bias² + Residual')
    plt.xlabel(param_name)
    plt.ylabel('Error')
    plt.legend()
    plt.title(f'Bias-Variance Trade-off vs {param_name}')
    plt.grid(True)
    plt.show() 


def plot_bias_variance_trade_off_lasso(
    param_range: list,
    errors: np.ndarray,
    variances: np.ndarray,
    bias_residuals: np.ndarray,
    param_name: str,
    log_scale: bool = True  # Nouveau paramètre
):
    """
    Visualise le compromis biais-variance en fonction d'un paramètre.
    
    Args:
        param_range: Liste des valeurs du paramètre
        errors: Tableau des erreurs
        variances: Tableau des variances
        bias_residuals: Tableau des biais² + résiduels
        param_name: Nom du paramètre (pour l'affichage)
        log_scale: Si True, utilise une échelle logarithmique pour l'axe x
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, errors, 'r-', label='Expected Error')
    plt.plot(param_range, variances, 'b-', label='Variance')
    plt.plot(param_range, bias_residuals, 'g-', label='Bias² + Residual')
    
    if log_scale:
        plt.xscale('log')  # Échelle logarithmique pour l'axe x
        
    plt.xlabel(param_name)
    plt.ylabel('Error')
    plt.legend()
    plt.title(f'Bias-Variance Trade-off vs {param_name}')
    plt.grid(True)
    plt.show()