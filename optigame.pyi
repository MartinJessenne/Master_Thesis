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

    @property
    def gaps_history(self) -> npt.NDArray[np.float64]:
        """Historique des duality gap de taille (num_iteration)"""

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
        
    # TODO: Ajouter la définition statique pour omwuoftrl
    @staticmethod
    def omuwoftrl(eta: float, dim: int) -> Optimizer:
        """Instancie OMWU - Optimistic Online Mirror Descent selon la méthode 
        Optismitic Follow the Regularized Leader."""

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

def neighborhood_exploration(
    p_lambda: npt.NDArray[np.float64], 
    q_gamma: npt.NDArray[np.float64], 
    optimizer: Optimizer, 
    num_steps: int, 
    normalize_matrix: bool
) -> list[GameResult]:
    """
    Explore le voisinage d'un jeu paramétré par des valeurs de lambda et gamma dans un array numpy de même dimension (num_explorations).
    
    Retourne une liste de GameResult contenant les historiques de convergence
    pour chaque point du voisinage évalué en parallèle par Rust.
    """
    ...