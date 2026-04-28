## Conceptual Logic

Pour confirmer ton hypothèse sur la vitesse de convergence Best-Iterate face à un mauvais conditionnement simultané ($M_δ$) par rapport à un mauvais conditionnement simple ($A_δ$), il y a trois étapes fondamentales à accomplir.

**1. Isoler la métrique "Best-Iterate"**
Actuellement, `analysis.py` affiche le Duality Gap de la dernière itération ("Last-Iterate"). Pour vérifier ton hypothèse, tu dois tracer le "Best-Iterate" Duality Gap, qui se définit mathématiquement comme le minimum cumulé de l'erreur jusqu'à l'itération $T$ : $\min_{s \le T} \text{Gap}(x_s, y_s)$.

**3. Matérialiser $A_δ$ et $M_δ$ et la Baseline Théorique**
Pour valider l'impact du conditionnement, tu dois instancier spécifiquement :
- $A_δ$ : une seule frontière approchée (e.g., $\lambda \to 0$, $\gamma$ fixe).
- $M_δ$ : un mauvais conditionnement simultané (e.g., $\lambda \to 0$ et $\gamma \to 0$ ensemble).
Enfin, tu devras superposer la baseline théorique de convergence $O(1/T)$ ou $O(1/\sqrt{T})$ sur ces graphes log-log pour vérifier visuellement si $M_δ$ dégrade le taux de Best-Iterate de façon plus sévère que la théorie et $A_δ$.

## API Reference Table

| Composant | Élément | Description et Usage |
| :--- | :--- | :--- |
| **Python / NumPy** | `np.minimum.accumulate(array)` | Fonction Numpy qui renvoie le minimum cumulatif d'un tableau. Parfait pour extraire la série temporelle du Best-Iterate à partir de ton `gaps_history`. |
| **Rust / optimizers.rs** | `step_x.map(...)` | Lors de la mise à jour OFTRL, il faut instancier le nouveau vecteur directement via `map` sur `step_x`, **sans** le multiplier par `state.x.as_array()`. |
| **Python / Matplotlib** | `ax.plot(x, y, label="O(1/T)")` | À utiliser pour générer la courbe de référence théorique $f(t) = \frac{C}{t}$ où $C$ est une constante d'échelle ajustée visuellement. |

## Logical Checklist

- [ ] Aller dans `src/optimizers.rs` à la méthode `step` pour `OmwuOftrl`.
- [ ] Retirer la multiplication par `state.x.as_array()` et `state.y.as_array()` lors de la définition de `x` et `y`.
- [ ] Recompiler le package via `maturin develop` ou en exécutant l'environnement marimo/python.
- [ ] Dans `analysis.py`, ajouter une cellule pour calculer le Best-Iterate : appliquer `np.minimum.accumulate` sur les historiques de gaps.
- [ ] Générer les jeux de test $A_δ$ (un bord) et $M_δ$ (deux bords) en paramétrant finement $\lambda$ et $\gamma$.
- [ ] Mettre à jour `neighborhood_exploration_plot` ou créer un nouveau plot pour superposer : le Best-Iterate de $A_δ$, le Best-Iterate de $M_δ$, et la courbe de référence $O(1/T)$.

## Structural Outline

# Fix in Rust (`src/optimizers.rs`)
// ... Compute max_step_x and max_step_y using the cumulative gradients ...
// Create new distribution without multiplying by state.x
// let mut x = step_x.map(|&s| f64::exp(s - max_step_x)); 
// let mut y = step_y.map(|&s| f64::exp(s - max_step_y));

# Computation in Python (`analysis.py`)
// Extract the standard gaps history
// gaps = result.gaps_history
// Compute Best Iterate sequence
// best_iterate_gaps = np.minimum.accumulate(gaps)

# Comparison plotting (`analysis.py`)
// Plot best_iterate_gaps for A_delta
// Plot best_iterate_gaps for M_delta
// Define theoretical baseline array: baseline = constant / np.arange(1, len(gaps) + 1)
// Plot baseline with a distinct line style

