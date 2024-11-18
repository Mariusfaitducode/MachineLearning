import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les résultats
results = np.load('results.npy', allow_pickle=True).item()

# Configuration du style des graphiques
# plt.style.use('seaborn')
# sns.set_palette("husl")

# Fonction pour créer un subplot avec un style cohérent
def plot_model_results(ax, params, errors, variances, bias_residuals, title, xlabel):
    ax.plot(params, errors, 'r-o', label='Erreur totale', linewidth=2)
    ax.plot(params, variances, 'b-^', label='Variance', linewidth=2)
    ax.plot(params, bias_residuals, 'g-s', label='Biais² + Résiduel', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Erreur')
    ax.grid(True)
    ax.legend()
    
    # Ajouter les valeurs sur les points
    for line in [errors, variances, bias_residuals]:
        for x, y in zip(params, line):
            ax.annotate(f'{y:.3f}', 
                       (x, y), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')

# Créer une figure avec 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# Plot k-NN results
plot_model_results(
    ax1,
    results['knn']['params'],
    results['knn']['errors'],
    results['knn']['variances'],
    results['knn']['bias_residuals'],
    'Compromis Biais-Variance pour k-NN',
    'Nombre de voisins (k)'
)

# Plot Lasso results
plot_model_results(
    ax2,
    results['lasso']['params'],
    results['lasso']['errors'],
    results['lasso']['variances'],
    results['lasso']['bias_residuals'],
    'Compromis Biais-Variance pour Lasso',
    'Alpha (échelle log)'
)
ax2.set_xscale('log')  # Échelle logarithmique pour alpha

# Plot Decision Tree results
plot_model_results(
    ax3,
    range(len(results['tree']['params'])),  # Utiliser des indices pour l'axe x
    results['tree']['errors'],
    results['tree']['variances'],
    results['tree']['bias_residuals'],
    'Compromis Biais-Variance pour Arbre de Décision',
    'Profondeur maximale'
)
# Remplacer les étiquettes de l'axe x par les vraies valeurs
ax3.set_xticks(range(len(results['tree']['params'])))
ax3.set_xticklabels([str(d) for d in results['tree']['params']])

plt.tight_layout()
plt.show()

# Créer un tableau comparatif des meilleures performances
print("\nMeilleures performances pour chaque modèle:")
print("-" * 60)
print(f"{'Modèle':<15} {'Meilleur paramètre':<20} {'Erreur minimale':<15}")
print("-" * 60)

# k-NN
best_knn_idx = np.argmin(results['knn']['errors'])
print(f"k-NN{' '*11} k={results['knn']['params'][best_knn_idx]:<18} {results['knn']['errors'][best_knn_idx]:.4f}")

# Lasso
best_lasso_idx = np.argmin(results['lasso']['errors'])
print(f"Lasso{' '*10} α={results['lasso']['params'][best_lasso_idx]:<18} {results['lasso']['errors'][best_lasso_idx]:.4f}")

# Decision Tree
best_tree_idx = np.argmin(results['tree']['errors'])
print(f"Arbre{' '*10} depth={results['tree']['params'][best_tree_idx]:<15} {results['tree']['errors'][best_tree_idx]:.4f}")
print("-" * 60) 