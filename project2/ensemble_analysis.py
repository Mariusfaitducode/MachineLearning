import numpy as np
from data import load_wine_quality
from bias_variance import estimate_bias_variance
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Chargement des données
X, y = load_wine_quality()

# Configuration
N_SAMPLES = 250
N_ITERATIONS = 100
n_estimators_range = [1, 5, 10, 20, 50, 100]
max_depths = [3, None]  # Test avec arbres contraints et non contraints

def analyze_ensemble_methods():
    results = {
        'bagging': {depth: {'errors': [], 'variances': [], 'bias_residuals': []} 
                   for depth in max_depths},
        'boosting': {depth: {'errors': [], 'variances': [], 'bias_residuals': []} 
                    for depth in max_depths}
    }
    
    for depth in max_depths:
        depth_str = str(depth) if depth else "None"
        print(f"\nAnalyse avec max_depth={depth_str}")
        
        # Test pour différents nombres d'estimateurs
        for n_est in n_estimators_range:
            print(f"  Nombre d'estimateurs: {n_est}")
            
            # Bagging
            bagging = BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=depth),
                n_estimators=n_est,
                random_state=42
            )
            error, variance, bias_res = estimate_bias_variance(
                X, y, bagging, n_samples=N_SAMPLES, n_iterations=N_ITERATIONS
            )
            results['bagging'][depth]['errors'].append(error)
            results['bagging'][depth]['variances'].append(variance)
            results['bagging'][depth]['bias_residuals'].append(bias_res)
            
            # Boosting
            boosting = GradientBoostingRegressor(
                max_depth=depth,
                n_estimators=n_est,
                learning_rate=0.1,
                random_state=42
            )
            error, variance, bias_res = estimate_bias_variance(
                X, y, boosting, n_samples=N_SAMPLES, n_iterations=N_ITERATIONS
            )
            results['boosting'][depth]['errors'].append(error)
            results['boosting'][depth]['variances'].append(variance)
            results['boosting'][depth]['bias_residuals'].append(bias_res)
    
    return results

# Exécution de l'analyse
results = analyze_ensemble_methods()

# Sauvegarde des résultats
np.save('ensemble_results.npy', results)

# Visualisation
def plot_ensemble_results(results, method, depth):

    # Paramètres utilisés dans l'analyse
    n_estimators_range = [1, 5, 10, 20, 50, 100]
    max_depths = [2, 3, None]
    N_SAMPLES = 250
    N_ITERATIONS = 100
    
    print(f"\nRésultats pour {method.capitalize()} (max_depth={depth}):")
    print("  n_estimators | Erreur    | Variance  | Biais² + Résiduel")
    print("  -------------|-----------|-----------|------------------")
    for i, n_est in enumerate(n_estimators_range):
        error = results[method][depth]['errors'][i]
        variance = results[method][depth]['variances'][i] 
        bias_res = results[method][depth]['bias_residuals'][i]
        print(f"  {n_est:>11} |    {error:.4f} |    {variance:.4f} |    {bias_res:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, results[method][depth]['errors'], 'r-', 
             label='Expected Error')
    plt.plot(n_estimators_range, results[method][depth]['variances'], 'b-', 
             label='Variance')
    plt.plot(n_estimators_range, results[method][depth]['bias_residuals'], 'g-', 
             label='Bias² + Residual')
    
    depth_str = str(depth) if depth else "None"
    plt.title(f'{method.capitalize()} - Depth {depth_str}')
    plt.xlabel("Nombre d'estimateurs")
    plt.ylabel('Erreur')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualisation pour chaque configuration
for method in ['bagging', 'boosting']:
    for depth in max_depths:
        plot_ensemble_results(results, method, depth) 