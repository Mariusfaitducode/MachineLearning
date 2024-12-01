import numpy as np
from data import load_wine_quality
from bias_variance import estimate_bias_variance, plot_bias_variance_trade_off
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# Chargement des données
X, y = load_wine_quality()

# Configuration
N_ITERATIONS = 100
sample_sizes = [50, 100, 250, 500, 1000, 2000]  # Différentes tailles d'échantillon

# Paramètres fixes pour k-NN et Lasso (choisis à partir des résultats précédents)
FIXED_K = 15  # Valeur optimale trouvée précédemment
FIXED_LAMBDA = 0.01  # Valeur optimale trouvée précédemment

def analyze_sample_size_impact():
    results = {
        'knn': {'errors': [], 'variances': [], 'bias_residuals': []},
        'lasso': {'errors': [], 'variances': [], 'bias_residuals': []},
        'tree_fixed': {'errors': [], 'variances': [], 'bias_residuals': []},
        'tree_full': {'errors': [], 'variances': [], 'bias_residuals': []}
    }
    
    for n_samples in sample_sizes:
        print(f"\nAnalyse pour taille d'échantillon = {n_samples}")
        
        # k-NN avec k fixé
        knn = KNeighborsRegressor(n_neighbors=FIXED_K)
        error, variance, bias_res = estimate_bias_variance(
            X, y, knn, n_samples=n_samples, n_iterations=N_ITERATIONS
        )
        results['knn']['errors'].append(error)
        results['knn']['variances'].append(variance)
        results['knn']['bias_residuals'].append(bias_res)
        
        # Lasso avec λ fixé
        lasso = Lasso(alpha=FIXED_LAMBDA, max_iter=10000)
        error, variance, bias_res = estimate_bias_variance(
            X, y, lasso, n_samples=n_samples, n_iterations=N_ITERATIONS
        )
        results['lasso']['errors'].append(error)
        results['lasso']['variances'].append(variance)
        results['lasso']['bias_residuals'].append(bias_res)
        
        # Arbre avec profondeur fixe (par exemple, 3)
        tree_fixed = DecisionTreeRegressor(max_depth=3)
        error, variance, bias_res = estimate_bias_variance(
            X, y, tree_fixed, n_samples=n_samples, n_iterations=N_ITERATIONS
        )
        results['tree_fixed']['errors'].append(error)
        results['tree_fixed']['variances'].append(variance)
        results['tree_fixed']['bias_residuals'].append(bias_res)
        
        # Arbre complet (sans limite de profondeur)
        tree_full = DecisionTreeRegressor(max_depth=None)
        error, variance, bias_res = estimate_bias_variance(
            X, y, tree_full, n_samples=n_samples, n_iterations=N_ITERATIONS
        )
        results['tree_full']['errors'].append(error)
        results['tree_full']['variances'].append(variance)
        results['tree_full']['bias_residuals'].append(bias_res)
    
    return results

# Exécution de l'analyse
results = analyze_sample_size_impact()

# Sauvegarde des résultats
np.save('sample_size_results.npy', results)

# Visualisation des résultats pour chaque modèle
def plot_sample_size_results(results, model_name):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, results[model_name]['errors'], 'r-', label='Expected Error')
    plt.plot(sample_sizes, results[model_name]['variances'], 'b-', label='Variance')
    plt.plot(sample_sizes, results[model_name]['bias_residuals'], 'g-', label='Bias² + Residual')
    plt.xlabel('Taille de l\'échantillon d\'apprentissage')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.legend()
    plt.title(f'Impact de la taille d\'échantillon - {model_name}')
    plt.grid(True)
    plt.show()

# Visualisation pour chaque modèle
for model in ['knn', 'lasso', 'tree_fixed', 'tree_full']:

    print(f"\nRésultats pour {model}:")
    print("  Taille échantillon | Erreur    | Variance  | Biais² + Résiduel")
    print("  --------------------|-----------|-----------|------------------")
    for i, size in enumerate(sample_sizes):
        print(f"  {size:>16d} | {results[model]['errors'][i]:9.4f} | "
              f"{results[model]['variances'][i]:9.4f} | "
              f"{results[model]['bias_residuals'][i]:9.4f}")
    plot_sample_size_results(results, model) 