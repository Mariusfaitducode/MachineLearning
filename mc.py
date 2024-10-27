import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from data import make_dataset


def tune_hyperparameters(X, y, model, param_grid):
    best_score = 0
    best_params = {}
    
    for param, values in param_grid.items():
        for value in values:
            model.set_params(**{param: value})
            scores = cross_val_score(model, X, y, cv=5)
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = {param: value}
    
    return best_params

def evaluate_model(X, y, model, n_iterations=5):
    accuracies = []
    
    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    return np.mean(accuracies), np.std(accuracies)

def compare_methods(n_iterations=5, n_irrelevant=0):
    # Generate dataset
    X, y = make_dataset(n_points=1000, n_irrelevant=n_irrelevant)
    
    # Define models and parameter grids
    models = {
        'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 7, 10, None]}),
        'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 11, 15]}),
        'Perceptron': (Perceptron(), {'eta0': [0.1, 0.01, 0.001, 0.0001]})
    }
    
    results = {}
    
    for name, (model, param_grid) in models.items():
        best_params = tune_hyperparameters(X, y, model, param_grid)
        model.set_params(**best_params)
        mean_accuracy, std_accuracy = evaluate_model(X, y, model, n_iterations)
        results[name] = {
            'best_params': best_params,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        }
    
    return results

# Compare methods without noisy features
results_without_noise = compare_methods()

# Compare methods with noisy features
results_with_noise = compare_methods(n_irrelevant=200)

# Print results
print("Results without noisy features:")
for name, result in results_without_noise.items():
    print(f"{name}:")
    print(f"  Best parameters: {result['best_params']}")
    print(f"  Mean accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")

print("\nResults with noisy features:")
for name, result in results_with_noise.items():
    print(f"{name}:")
    print(f"  Best parameters: {result['best_params']}")
    print(f"  Mean accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
