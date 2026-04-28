# Stratégies de Visualisation : Quantifier le Chaos du "Last Iterate"

## Conceptual Logic
Dans les jeux à  somme nulle, les trajectoires peuvent orbiter violemment ou diverger, et lisser la courbe revient à  cacher ce phénomène sous le tapis pour faire ressembler le "last-iterate" à  une convergence en moyenne ("average-iterate").

L'objectif de cette nouvelle stratégie est de **compresser l'entièreté de la trajectoire (ou sa partie asymptotique) en un seul scalaire** qui caractérise la sévérité du chaos ou de la non-convergence, pour chaque configuration expérimentale (ex. pour chaque paire $\lambda, \gamma$). 

Voici quelques métriques mathématiques pour quantifier ce comportement :
1. **Maximum sur les 10% dernières itérations** (L'idée de ton tuteur) : Cela permet d'ignorer la phase d'initialisation (burn-in) et de voir si l'algorithme a fini par se stabiliser ou s'il oscille toujours avec une grande amplitude à  la fin.
2. **Variance Globale (ou sur la fin)** : Une variance élevée indique que le gap fait des bonds énormes, signe d'une grande instabilité.
3. **Variation Totale (Total Variation)** : C'est la somme des différences absolues entre itérations consécutives ($\sum |gap_t - gap_{t-1}|$). C'est une excellente métrique mathématique pour mesurer la "quantité d'oscillations". Une courbe lisse aura une faible variation totale, tandis qu'une courbe en dents de scie aura une variation totale très élevée.

En utilisant l'interface interactive de `marimo`, tu pourras créer un menu déroulant pour basculer instantanément d'une métrique à  l'autre et observer comment l'espace des hyperparamètres réagit à  ces différentes définitions du "chaos".

## API Reference Table

| Librairie / Outil | Méthode / Classe | Description et Usage |
| :--- | :--- | :--- |
| **NumPy** | `np.var(array)` | Calcule la variance d'un tableau, utile pour mesurer la dispersion des valeurs de gap. |
| **NumPy** | `np.max(array)` | Trouve la valeur maximale dans un tableau ou un sous-tableau. |
| **NumPy** | `np.diff(array)` | Calcule la différence entre les éléments consécutifs d'un tableau ($a[i+1] - a[i]$). Idéal pour la Variation Totale. |
| **NumPy** | `np.abs(array)` | Applique la valeur absolue. Combiné avec `np.sum()` et `np.diff()`, il donne l'amplitude totale des oscillations. |
| **Marimo** | `mo.ui.dropdown(options, value)` | Crée un widget de sélection interactif dans ton notebook. `options` est un dictionnaire de choix `{Label: Valeur}`. |

## Logical Checklist

- [ ] **Définir la fonction d'extraction de métriques** : Créer une fonction `compute_chaos_metric(gaps_history, metric_type)` qui prend l'historique complet et renvoie un scalaire.
- [ ] **Implémenter le slicing asymptotique** : Pour la métrique de ton tuteur, isoler les 10% dernières itérations en utilisant le slicing Python (ex: `gaps[-taille_fenetre:]`).
- [ ] **Implémenter la Variation Totale** : Calculer la somme des valeurs absolues des différences consécutives pour quantifier l'effet "dents de scie".
- [ ] **Intégrer le widget Marimo** : Déclarer un `mo.ui.dropdown` contenant les différentes métriques ("Max 10% finaux", "Variance", "Variation Totale", etc.).
- [ ] **Lier l'UI au graphique** : Passer la valeur sélectionnée par le dropdown (`dropdown.value`) à  ta fonction de plotting pour qu'elle recalcule et réaffiche les résultats dynamiquement.

## Structural Outline

```python
# 1. Définition de la fonction de calcul de métrique
# def compute_chaos_metric(gaps_history, metric_type="max_last_10"):
#     n_iters = len(gaps_history)
#     last_10_percent = gaps_history[-int(0.1 * n_iters):]
#     
#     if metric_type == "max_last_10":
#         # Implémenter le max sur la fenêtre finale
#         return np.max(last_10_percent)
#     
#     elif metric_type == "variance":
#         # Implémenter la variance
#         return np.var(gaps_history)
#     
#     elif metric_type == "total_variation":
#         # Implémenter la Variation Totale pour mesurer l'oscillation
#         # return np.sum(np.abs(np.diff(gaps_history)))
#         pass
#         
#     else:
#         raise ValueError("Métrique inconnue")

# 2. Dans une cellule Marimo, création du menu interactif
# metric_dropdown = mo.ui.dropdown(
#     options={
#         "Max des 10% finaux": "max_last_10",
#         "Variance Globale": "variance",
#         "Variation Totale (Oscillations)": "total_variation"
#     },
#     value="max_last_10",
#     label="Sélectionner la métrique de Chaos :"
# )
# # Ne pas oublier de renvoyer metric_dropdown à  la fin de la cellule pour l'afficher

# 3. Dans la cellule de plotting, utiliser la valeur du widget
# metric_choisie = metric_dropdown.value
# Pour chaque run (pour chaque lambda/gamma) :
#     scalaire = compute_chaos_metric(run.gaps, metric_choisie)
# ... utiliser ces scalaires pour colorer ou plotter tes points ...
```

