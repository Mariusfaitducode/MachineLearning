import numpy as np
from data import load_wine_quality
from bias_variance import evaluate_model_complexity, plot_bias_variance_trade_off, plot_bias_variance_trade_off_lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# 1. Loading data
print("Loading data...")
X, y = load_wine_quality()

# 2. General parameters configuration
N_SAMPLES = 250  # Learning sample size
N_ITERATIONS = 100  # Number of iterations for estimation

# 3. k-NN analysis
print("\n=== k-NN Analysis ===")
def create_knn(k):
    return KNeighborsRegressor(n_neighbors=k)

k_range = [1, 3, 5, 7, 11, 15, 21, 31, 51]
knn_errors, knn_variances, knn_bias_residuals = evaluate_model_complexity(
    X, y,
    create_knn,
    k_range,
    'k (n_neighbors)',
    n_samples=N_SAMPLES,
    n_iterations=N_ITERATIONS
)

plot_bias_variance_trade_off(
    k_range,
    knn_errors,
    knn_variances,
    knn_bias_residuals,
    'Number of neighbors (k)'
)

# 4. Lasso analysis
print("\n=== Lasso Analysis ===")
def create_lasso(alpha):
    return Lasso(alpha=alpha, max_iter=10000)

alpha_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
lasso_errors, lasso_variances, lasso_bias_residuals = evaluate_model_complexity(
    X, y,
    create_lasso,
    alpha_range,
    'alpha',
    n_samples=N_SAMPLES,
    n_iterations=N_ITERATIONS
)

plot_bias_variance_trade_off_lasso(
    alpha_range,
    lasso_errors,
    lasso_variances,
    lasso_bias_residuals,
    'Regularization parameter (alpha)'
)

# 5. Decision trees analysis
print("\n=== Decision Trees Analysis ===")
def create_tree(max_depth):
    return DecisionTreeRegressor(max_depth=max_depth)

depth_range = [1, 2, 3, 5, 7, 10, 15, 20, None]
tree_errors, tree_variances, tree_bias_residuals = evaluate_model_complexity(
    X, y,
    create_tree,
    depth_range,
    'max_depth',
    n_samples=N_SAMPLES,
    n_iterations=N_ITERATIONS
)

plot_bias_variance_trade_off(
    [str(d) if d is not None else 'None' for d in depth_range],
    tree_errors,
    tree_variances,
    tree_bias_residuals,
    'Maximum depth'
)

# 6. Saving results
print("\n=== Saving Results ===")
results = {
    'knn': {
        'params': k_range,
        'errors': knn_errors,
        'variances': knn_variances,
        'bias_residuals': knn_bias_residuals
    },
    'lasso': {
        'params': alpha_range,
        'errors': lasso_errors,
        'variances': lasso_variances,
        'bias_residuals': lasso_bias_residuals
    },
    'tree': {
        'params': depth_range,
        'errors': tree_errors,
        'variances': tree_variances,
        'bias_residuals': tree_bias_residuals
    }
}

np.save('results.npy', results)
print("Results saved in 'results.npy'") 