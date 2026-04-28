---
type: Logic
status: Closed
related_pillar: "[[Ch3_Methodology]]"
tags: [thesis, chapter_3, documentation, rust, pyo3, python]
---
# Documentation Python pour la Librairie Rust (PyO3)

## Conceptual Logic
Lorsque tu compiles du code Rust en un module Python avec PyO3 et Maturin, le résultat est un fichier binaire (un `.pyd` sous Windows ou un `.so` sous Linux/Mac). Les éditeurs de code Python (comme VSCode avec Pylance ou Pyright) ne peuvent pas "lire" ce code source binaire pour deviner les types et les docstrings des entités exportées (classes, fonctions, énumérations, etc.).

La solution canonique en Python est de créer un **Stub File** (fichier d'en-tête ou de typage), portant l'extension `.pyi`. Ce fichier contient uniquement la structure des objets exportés, les signatures avec les annotations de type (type hints) et les docstrings, mais sans aucune implémentation (le corps est remplacé par des points de suspension `...`). 

En plaçant un fichier `optigame.pyi` à la racine de ton projet (là où s'exécute ton code Python), ton éditeur saura l'utiliser comme une "carte de référence" pour te fournir l'autocomplétion, la vérification des types et la documentation interactive. 

Cette méthode s'applique à **tout type** d'objet exposé par PyO3 :
- **Classes (`#[pyclass]`)** : Définies avec le mot clé `class`.
- **Fonctions (`#[pyfunction]`)** : Définies avec le mot clé `def` au niveau global du module (en dehors de toute classe).
- **Énumérations (Enums)** : Exposées ou redéfinies en Python en héritant de `enum.Enum` ou en utilisant des classes simples.
- **Types Personnalisés (Custom Types/Aliases)** : Définis avec un alias classique (ex: `MyType = list[int]`) pour clarifier les signatures.

## API Reference Table
Voici les principales entités Rust de ton projet (`experiments.rs` et `optimizers.rs`) qu'il faut documenter :

| Entité Rust                           | Équivalent Type Python               | Description à documenter                                                                                           |
| :------------------------------------ | :----------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `GameState` (Class)                   | `class GameState:`                   | État du jeu. Accepte des tableaux numpy unidimensionnels (x, y) et bidimensionnel (a).                             |
| `GameResult` (Class)                  | `class GameResult:`                  | Contient les propriétés `x_history`, `y_history` (matrices 2D) et `gaps_history` (vecteur 1D).                     |
| `Optimizer` (Class)                   | `class Optimizer:`                   | Factory exposant les méthodes statiques `@staticmethod` : `ogda`, `omwuoomd` et `omwuoftrl`.                       |
| `Experiment` (Class)                  | `class Experiment:`                  | Prend un `GameState`, un `Optimizer` et un entier `num_steps`. Expose `run_experiment_until_convergence_in_place`. |
| `neighborhood_exploration` (Function) | `def neighborhood_exploration(...):` | Fonction globale qui prend des tableaux lambda/gamma et retourne une liste de `GameResult` (`list[GameResult]`).   |

## Logical Checklist
- [ ] **Déclarer les classes et propriétés** : Structurer chaque classe avec `class`, définir `__init__`, et utiliser `@property` pour les getters.
- [ ] **Déclarer les fonctions globales** : Définir les fonctions exposées par `#[pyfunction]` directement à la racine du fichier `.pyi` avec `def`.
- [ ] **Importer les types nécessaires** : Importer `numpy.typing` pour les tableaux, ou le module `enum` si tu exposes des énumérations.
- [ ] **Ajouter la documentation** : Rédiger tes docstrings entre triple-guillemets (`"""..."""`) sous la signature de chaque fonction ou classe.
- [ ] **Mettre `...` comme implémentation** : Assure-toi qu'aucune méthode ou fonction n'a de logique ; elles doivent toutes finir par `...`.

## Structural Outline
Voici le squelette générique pour ton fichier `optigame.pyi`, te montrant comment typer différents concepts, y compris la fonction globale `neighborhood_exploration`.

```python
# Fichier: optigame.pyi
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum

class GameState:
    """Représente l'état courant d'un jeu à somme nulle."""
    def __init__(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], matrix: npt.NDArray[np.float64]) -> None:
        ...
        
    @property
    def a(self) -> npt.NDArray[np.float64]:
        ...

class GameResult:
    """Stocke l'historique complet d'une expérience."""
    @property
    def x_history(self) -> npt.NDArray[np.float64]:
        ...

class Optimizer:
    """Factory pour générer les algorithmes d'optimisation."""
    @staticmethod
    def ogda(eta: float, dim: int) -> 'Optimizer':
        ...

class Experiment:
    """Orchestre la simulation d'optimisation d'un jeu."""
    def __init__(self, state: GameState, optimizer: Optimizer, num_steps: int) -> None:
        ...

def neighborhood_exploration(
    p_lambda: npt.NDArray[np.float64], 
    q_gamma: npt.NDArray[np.float64], 
    optimizer: Optimizer, 
    num_steps: int, 
    normalize_matrix: bool
) -> list[GameResult]:
    """
    Explore le voisinage d'un jeu paramétré par des vecteurs lambda et gamma.
    """
    ...
```
