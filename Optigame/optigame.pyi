import numpy as np
import numpy.typing as npt

class GameState:
    """
    Représente l'état courant d'un jeu à somme nulle.
    """
    def __init__(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], matrix: npt.NDArray[np.float64]) -> GameState:
        """Initialise l'état avec les stratégies de départ (x, y) et la matrice des gains (matrix)."""
        ...
        
    @property
    def a(self) -> npt.NDArray[np.float64]:
        """Retourne la matrice des gains."""
        ...

class GameResult:
    """
    Stocke l'historique complet d'une expérience (probabilités et duality gap).
    """
    @property
    def x_history(self) -> npt.NDArray[np.float64]:
        """Historique de la stratégie X de taille (num_steps, dim)."""
        ...
        
    @property
    def y_history(self) -> npt.NDArray[np.float64]:
        """Historique de la stratégie Y de taille (num_steps, dim)."""
        ...

    @property
    def gaps_history(self) -> npt.NDArray[np.float64]:
        """Historique des duality gap de taille (num_iteration)"""
        ...

class Ogda:
    """
    Optimistic Gradient Descent Ascent (OGDA) optimizer.
    """
    def __init__(self, eta: float, dim: int) -> None:
        """Initialise l'optimiseur OGDA."""
        ...

class OmwuOomd:
    """
    Optimistic Multiplicative Weights Update via Optimistic Online Mirror Descent.
    """
    def __init__(self, eta: float, dim: int) -> None:
        """Initialise l'optimiseur OMWU (OOMD)."""
        ...

class OmwuOftrl:
    """
    Optimistic Multiplicative Weights Update via Optimistic Follow the Regularized Leader.
    """
    def __init__(self, eta: float, dim: int) -> None:
        """Initialise l'optimiseur OMWU (OFTRL)."""
        ...

class Optimizer:
    """
    Factory pour générer les algorithmes d'optimisation.
    """
    @staticmethod
    def ogda(eta: float, dim: int) -> Optimizer:
        """Instancie l'Optimistic Gradient Descent Ascent."""
        ...
        
    @staticmethod
    def omwuoomd(eta: float, dim: int) -> Optimizer:
        """Instancie OMWU - Optimistic Online Mirror Descent selon la méthode
        Optimistic Online Mirror Descent."""
        ...
        
    @staticmethod
    def omwuoftrl(eta: float, dim: int) -> Optimizer:
        """Instancie OMWU - Optimistic Online Mirror Descent selon la méthode 
        Optimistic Follow the Regularized Leader."""
        ...

class Experiment:
    """
    Orchestre la simulation d'optimisation d'un jeu.
    """
    def __init__(self, state: GameState, optimizer: Optimizer, num_steps: int) -> Experiment:
        ...
        
    def run_experiment_until_convergence_in_place(self) -> GameResult:
        """
        Exécute l'expérience de manière in-place (modifie le GameState fourni).
        S'arrête dès que le duality gap est < 10e-9 ou à la fin de num_steps.
        """
        ...

class PyConcentricOutput:
    """
    Résultat d'une exploration concentrique.
    """
    @property
    def slice_boundaries(self) -> npt.NDArray[np.float64]:
        """Limites de chaque tranche de taille (num_slices, 2)."""
        ...

    @property
    def metrics(self) -> npt.NDArray[np.float64]:
        """Métriques de convergence pour chaque tranche de taille (num_slices, runs_per_slice)."""
        ...

class PyScatteredOutput:
    """
    Résultat d'une exploration dispersée (scattered).
    """
    @property
    def norms(self) -> npt.NDArray[np.float64]:
        """Normes de perturbation pour chaque exploration de taille (num_exploration)."""
        ...

    @property
    def metrics(self) -> npt.NDArray[np.float64]:
        """Métriques de convergence pour chaque exploration de taille (num_exploration)."""
        ...

def neighborhood_exploration(
    matrices: npt.NDArray[np.float64], 
    optimizer: Optimizer, 
    num_steps: int, 
    normalize_matrix: bool
) -> list[GameResult]:
    """
    Explore un ensemble de jeux définis par une série de matrices 2x2.
    `matrices` est un array numpy 3D de forme (num_explorations, 2, 2).
    
    Retourne une liste de GameResult contenant les historiques de convergence
    pour chaque matrice évaluée en parallèle par Rust.
    """
    ...

def concentric_exploration(
    a_delta: npt.NDArray[np.float64],
    optimizer: Optimizer,
    num_exploration: int,
    num_steps: int,
    inner_radius: float,
    outer_radius: float,
    num_slices: int,
    metric_method: str,
    cutoff: float = 0.1,
) -> PyConcentricOutput:
    """
    Explore le voisinage d'une matrice par tranches concentriques.
    `a_delta` est un array numpy 2D.
    `optimizer` est l'algorithme d'optimisation à utiliser.
    `num_exploration` est le nombre total de simulations.
    `num_steps` est le nombre d'itérations par simulation.
    `inner_radius` et `outer_radius` définissent le rayon de perturbation.
    `num_slices` est le nombre de tranches concentriques.
    `metric_method` est la méthode de calcul de métrique ("max_last", "var_last", "total_var").
    `cutoff` est la fraction finale d'itérations utilisée (défaut: 0.1).
    """
    ...

def scattered_exploration(
    a_delta: npt.NDArray[np.float64],
    optimizer: Optimizer,
    num_exploration: int,
    num_steps: int,
    inner_radius: float,
    outer_radius: float,
    norm_str: str,
    metric_method: str,
    cutoff: float = 0.1,
) -> PyScatteredOutput:
    """
    Explore le voisinage d'une matrice de manière dispersée (scattered).
    `a_delta` est un array numpy 2D.
    `optimizer` est l'algorithme d'optimisation à utiliser.
    `num_exploration` est le nombre total de simulations.
    `num_steps` est le nombre d'itérations par simulation.
    `inner_radius` et `outer_radius` définissent le rayon de perturbation.
    `norm_str` spécifie la norme de perturbation à utiliser ('max', 'infinity', 'frobenius').
    `metric_method` est la méthode de calcul de métrique ("max_last", "var_last", "total_var").
    `cutoff` est la fraction finale d'itérations utilisée (défaut: 0.1).
    """
    ...
