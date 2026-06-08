import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import optigame
from typing import Callable

def generate_matrices(P_lambdas, Q_gammas, method="A_lambda_gamma"):
    """
    Goal: Build a 3D numpy array of shape (len(P_lambdas), 2, 2) containing
    the game matrices for each lambda/gamma pair.
    """
    # 1. Initialize a numpy array `matrices` of zeros with shape (len(P_lambdas), 2, 2)
    matrices = np.zeros((len(P_lambdas), 2, 2))
    
    # 2. Iterate over the length of P_lambdas
    for i in range(len(P_lambdas)):
        p = P_lambdas[i]
        q = Q_gammas[i]
        
        if method == "A_lambda_gamma":
            S = 1
            a = 1 + S*(1 - p - q)
            b = 1 - q*S 
            c = 1 - p*S
            d = 1
            matrix= np.array([[a, b], [c, d]], dtype=np.float64)
            matrices[i] = matrix
            
        elif method == "lemma5":
            # The paper defines NE as x* = (1-delta_x, delta_x). 
            # Since p is the first component, delta_x = 1 - p.
            # Similarly, delta_y = 1 - q.
            delta_x = 1 - p
            delta_y = 1 - q
            
            denom = 1 - delta_x if (1 - delta_x) != 0 else 1e-9
            # Implement the Lemma 5 matrix formulas
            a = (1 - delta_y) / denom
            b = (1 - delta_x - delta_y) / denom
            c = 0
            d = 1
            matrix = np.array([[a, b], [c, d]], dtype=np.float64)
            matrices[i] = matrix 
    # 3. Return the `matrices` array
    return matrices


class NeighborhoodExplorationResult:
    """
    Struct to hold the results of the neighborhood exploration computation,
    ready to be plotted by neighborhood_exploration_plot.
    """
    def __init__(self, list_of_results, x_axis, P_lambdas, Q_gammas):
        self.list_of_results = list_of_results
        self.x_axis= x_axis
        self.P_lambdas = P_lambdas
        self.Q_gammas = Q_gammas

def python_neighborhood_exploration(P_lambdas, Q_gammas, Optimizer, num_steps, Normalize_Matrix):
    x = np.array([0,0], dtype=np.float64)
    y = np.array([0,0], dtype=np.float64)

    list_of_results = []
    
    # TODO: Refactor loop to use generate_matrices instead of building matrices here
    matrices = generate_matrices(P_lambdas, Q_gammas)
    for matrix in matrices:
        g = optigame.GameState(x, y, matrix)
        e = optigame.Experiment(g, Optimizer, num_steps)
        result = e.run_experiment_until_convergence_in_place()
        list_of_results.append(result)

    return list_of_results

class GameOptimizer:
    def __init__(self, x_init, y_init, A, eta):
        # We assume x_init, y_init, A are already floats/mpf, handled externally to avoid copying the mpmath conversions
        self.x = x_init.copy()
        self.y = y_init.copy()
        self.x_hat = self.x.copy()
        self.y_hat = self.y.copy()
        self.A = A
        self.eta = eta
        self.history_x = [self.x.copy()]
        self.history_y = [self.y.copy()]

        val_max_y = np.max(self.x @ self.A) 
        val_min_x = np.min(self.A @ self.y)
        self.gaps = [val_max_y - val_min_x]

    def _compute_gradients(self, x, y):
        grad_x = self.A @ y
        grad_y = -self.A.T @ x 
        return grad_x, grad_y

def projection_simplex(v : np.ndarray, z=1):
    n_features: int = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class OGDA(GameOptimizer):
    def step(self):
        grad_x, grad_y = self._compute_gradients(self.x, self.y)
        self.x_hat = projection_simplex(self.x_hat - self.eta * grad_x)
        self.y_hat = projection_simplex(self.y_hat - self.eta * grad_y)
        self.x = projection_simplex(self.x_hat - self.eta * grad_x)
        self.y = projection_simplex(self.y_hat - self.eta * grad_y)

        self.history_x.append(self.x.copy())
        self.history_y.append(self.y.copy())
        val_max_y = np.max(self.x @ self.A) 
        val_min_x = np.min(self.A @ self.y)
        self.gaps.append(val_max_y - val_min_x)

class OMWU(GameOptimizer):
    def step(self):
        grad_x, grad_y = self._compute_gradients(self.x, self.y)

        x_hat_unnorm = self.x_hat * np.exp(-self.eta * grad_x)
        self.x_hat = x_hat_unnorm / np.sum(x_hat_unnorm)

        y_hat_unnorm = self.y_hat * np.exp(-self.eta * grad_y)
        self.y_hat = y_hat_unnorm / np.sum(y_hat_unnorm)

        x_unnorm = self.x_hat * np.exp(-self.eta * grad_x)
        self.x = x_unnorm / np.sum(x_unnorm)

        y_unnorm = self.y_hat * np.exp(-self.eta * grad_y)
        self.y = y_unnorm / np.sum(y_unnorm)

        self.history_x.append(self.x.copy())
        self.history_y.append(self.y.copy())
        val_max_y = np.max(self.x @ self.A) 
        val_min_x = np.min(self.A @ self.y)
        self.gaps.append(val_max_y - val_min_x)


class PythonGameResult:
    def __init__(self, x_history, y_history, gaps_history):
        self.x_history = x_history
        self.y_history = y_history
        self.gaps_history = gaps_history


def full_python_neighborhood_exploration(
    P_lambdas,
    Q_gammas,
    Optimizer_name : str = "Ogda",
    num_steps : int = 10_000,
    Normalize_Matrix : bool = True,
) -> list[PythonGameResult]:

    list_of_results = []

    for p_lambda, q_gamma in zip(P_lambdas, Q_gammas):
        S = 1.0
        a = 1.0 + S * (1.0 - p_lambda - q_gamma)
        b = 1.0 - q_gamma * S 
        c = 1.0 - p_lambda * S
        d = 1.0
        matrix = np.array([[a, b], [c, d]], dtype=np.float64)

        if Normalize_Matrix:
            m = np.min(matrix)
            M = np.max(matrix)
            if M - m > 1e-9:
                matrix = (matrix - m) / (M - m)

        x_init = np.array([0.5, 0.5])
        y_init = np.array([0.5, 0.5])

        eta = 0.1

        if Optimizer_name.lower() == "ogda":
            optimizer = OGDA(x_init, y_init, matrix, eta)
        elif Optimizer_name.lower() == "omwu":
            optimizer = OMWU(x_init, y_init, matrix, eta)
        else:
            raise ValueError(f"Unknown optimizer name: {Optimizer_name}")

        for _ in range(num_steps):
            optimizer.step()

        res = PythonGameResult(
            x_history=np.array(optimizer.history_x, dtype=float),
            y_history=np.array(optimizer.history_y, dtype=float),
            gaps_history=np.array(optimizer.gaps, dtype=float)
        )

        list_of_results.append(res)

    return list_of_results

def neighborhood_exploration_compute(
    Optimizer: optigame.Optimizer,
    Normalize_Matrix: bool =True,
    p_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x**2,
    q_transform: Callable[[np.ndarray], np.ndarray] = lambda x: 0.5 + x,
    execution_mode: str = 'full_rust',
    number_of_points: int = 500,
    num_steps: int = 10_000,
    method: str = "A_lambda_gamma"
    ) -> NeighborhoodExplorationResult:
    """
    pass closure arguments like this :

    def make_p(alpha):
        def p_fn(lmb):
            return lmb**alpha
        return p_fn

    def make_q(offset, beta):
        def q_fn(gam):
            return offset + gam**beta
        return q_fn
    """

    x_axis= np.linspace(0, 1, number_of_points)


    list_of_results = []

    P_lambdas = p_transform(x_axis) # First component of theoretical NE for x player
    Q_gammas = q_transform(x_axis) # Second component of theoretical NE for y player

    # Call rust function
    if execution_mode == 'full_rust':
        matrices = generate_matrices(P_lambdas, Q_gammas, method=method)
        list_of_results = optigame.neighborhood_exploration(matrices, Optimizer, num_steps, Normalize_Matrix)
    elif execution_mode == 'mixed_rust':
        list_of_results = python_neighborhood_exploration(P_lambdas, Q_gammas, Optimizer,num_steps, Normalize_Matrix)
    elif execution_mode == 'python':
        list_of_results = full_python_neighborhood_exploration(P_lambdas, Q_gammas, "Ogda", num_steps, Normalize_Matrix)
    else:
        raise ValueError(f"Invalid execution mode: {execution_mode}")

    return NeighborhoodExplorationResult(
        list_of_results=list_of_results,
        x_axis=x_axis,
        P_lambdas=P_lambdas,
        Q_gammas=Q_gammas
    )
    

def plot_list_of_results(list_of_results, x_axis, P_lambdas, Q_gammas, metric_type='max_last_10'):
    fig, (ax_last_i, ax_random_i, ax_best_i) = plt.subplots(figsize=(8, 15), nrows=3, ncols=1)
    metrics_last_it = []
    metrics_random_it = []
    metrics_best_it = []

    nb_it = len(list_of_results[0].gaps_history)

    for _result in list_of_results:
        last_i = _result.gaps_history

        best_i      = np.minimum.accumulate(_result.gaps_history)
        cumulative_sum     = np.cumsum(_result.gaps_history)
        counts             = np.arange(1, nb_it + 1)
        random_i    = cumulative_sum / counts



        if metric_type == 'max_last_10':
            # Maximum gap over the last 10% iterations for this lambda.
            metric_last_it = np.max(last_i[-int(0.1 * nb_it):])
            metric_random_it = np.max(random_i[-int(0.1 * nb_it):])
            metric_best_it = np.max(best_i[-int(0.1 * nb_it):])

        elif metric_type == 'var_last_10':
            # Variance of the last 10% iterations for this lambda.
            metric_last_it = np.var(last_i[-int(0.1 * nb_it):])
            metric_random_it = np.var(random_i[-int(0.1 * nb_it):])
            metric_best_it = np.var(best_i[-int(0.1 * nb_it):])

        elif metric_type == 'total_var':
            # Total variation of the full gap trajectory for this lambda.
            metric_last_it = np.sum(np.abs(np.diff(last_i)))
            metric_random_it = np.sum(np.abs(np.diff(random_i)))
            metric_best_it = np.sum(np.abs(np.diff(best_i)))

        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        metrics_last_it.append(metric_last_it)
        metrics_random_it.append(metric_random_it)
        metrics_best_it.append(metric_best_it)

    metrics_last_it = np.asarray(metrics_last_it)
    metrics_random_it = np.asarray(metrics_random_it)
    metrics_best_it = np.asarray(metrics_best_it)

    if len(x_axis) != len(metrics_last_it):
        raise ValueError(f"Expected as many metrics as x_axis values, got {len(metrics_last_it)} and {len(x_axis)}")

    ax_last_i.scatter(x_axis, metrics_last_it, color='tab:blue', alpha=0.5, label='Last Iterate')
    ax_random_i.scatter(x_axis, metrics_random_it, color='tab:blue', alpha=0.5, label='Random Iterate')
    ax_best_i.scatter(x_axis, metrics_best_it, color='tab:blue', alpha=0.5, label='Best Iterate')

    y_label = metric_type.replace('_', ' ').title() + '% \n Iterations Duality Gap Value'
    for ax in (ax_last_i, ax_random_i, ax_best_i):
        ax.set_xlabel('x')
        ax.set_ylabel(y_label)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(loc='upper left')
        ax.grid(True, ls="--", alpha=0.6)

        ax2 = ax.twinx()
        ax2.plot(x_axis, P_lambdas, color='tab:orange', linestyle='--', label=r'Theoretical $x^*_0$')
        ax2.plot(x_axis, Q_gammas, color='tab:green', linestyle='--', label=r'Theoretical $y^*_0$')
        ax2.set_ylabel('Nash Equilibrium')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_computed_vs_theoretical_NE(list_of_results, model_string, x_axis, P_lambdas, Q_gammas):
    fig, (ax_p_vs_NE, ax_q_vs_NE) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    final_x = []
    final_y = []
    for res in list_of_results:
        # Find the last index before zero-padding
        idx = np.where(res.gaps_history > 0)[0][-1] if np.any(res.gaps_history > 0) else 0
        final_x.append(res.x_history[idx])
        final_y.append(res.y_history[idx])  

    final_x = np.array(final_x)
    final_y = np.array(final_y)

    # Left plot: x_0 (p) player convergence
    ax_p_vs_NE.plot(x_axis, P_lambdas, label=r'Theoretical $x^*_0$', color='blue', linestyle='--')
    ax_p_vs_NE.plot(x_axis, final_x[:, 0], label=r'Computed Last-Iterate', color='blue', linestyle='-')
    ax_p_vs_NE.set_xlabel(r'Parameter $x$')
    ax_p_vs_NE.set_ylabel(r'Strategy Probability $p$')
    ax_p_vs_NE.set_title(r'Strategy $x^*_0$ Convergence')
    ax_p_vs_NE.legend(loc='lower left')
    ax_p_vs_NE.grid(True, ls="--", alpha=0.6)

    # Right plot: y_0 (q) player convergence
    ax_q_vs_NE.plot(x_axis, Q_gammas, label=r'Theoretical $y^*_0$', color='blue', linestyle='--')
    ax_q_vs_NE.plot(x_axis, final_y[:, 0], label=r'Computed Last-Iterate', color='blue', linestyle='-')
    ax_q_vs_NE.set_xlabel(r'Parameter $x$')
    ax_q_vs_NE.set_ylabel(r'Strategy Probability $q$')
    ax_q_vs_NE.set_title(r'Strategy $y^*_0$ Convergence')
    ax_q_vs_NE.legend(loc='lower left')
    ax_q_vs_NE.grid(True, ls="--", alpha=0.6)

    # fig.savefig(f"../images/{model_string}_Last_xo_vs_ne.svg", format="svg", dpi=300)
    
    plt.show()


def neighborhood_exploration_plot(NeighborhoodExplorationResult: NeighborhoodExplorationResult, model_string: str, metric_type='max_last_10'):

    list_of_results = NeighborhoodExplorationResult.list_of_results
    x_axis= NeighborhoodExplorationResult.x_axis
    P_lambdas = NeighborhoodExplorationResult.P_lambdas
    Q_gammas = NeighborhoodExplorationResult.Q_gammas

    plot_list_of_results(list_of_results, x_axis, P_lambdas, Q_gammas, metric_type=metric_type)

    plot_computed_vs_theoretical_NE(list_of_results, model_string, x_axis=x_axis, P_lambdas=P_lambdas, Q_gammas=Q_gammas)


def make_circular(center_x, center_y, r, theta):
    """
    This function takes as input center_x, center_y, r, theta,
    and returns two functions p_fn and q_fn that take as input x ranging [0, 1] and return the values of the Nash Equilibrium for point x^*_0 = center_x + r*cos(theta*x) and y^*_0 = center_y + r*sin(theta*x), 
    for this specific iteration. 
    """
    def p_fn(x):
        return center_x + r * np.cos(theta * x)

    def q_fn(x):
        return center_y + r * np.sin(theta * x)

    return p_fn, q_fn


def plot_2d_profile(results_OMWU, idx, ax, p_fn, q_fn, circular_delta):
    result = results_OMWU.list_of_results[idx]
    x_history = result.x_history
    y_history = result.y_history
    ax.set_title("2D Profile of the strategies")
    x_axis = np.linspace(0, 1, len(results_OMWU.list_of_results))
    (x_0_star, y_0_star) = p_fn(x_axis[idx]), q_fn(x_axis[idx])

    # Vectorized plotting
    ax.plot(x_history[:, 0], y_history[:, 0], '*', color='blue', alpha=0.5, label='Strategy Trajectory')

    # Plot markers once outside the loop
    ax.plot(0.5, 0.5, 'X', color='red', markersize=10, label='Starting Point')
    ax.plot(x_0_star, y_0_star, 'o', markersize=10, label='Iteration Nash Equilibrium', color='green')

    # Plot the baseline A_delta Nash Equilibrium (the center of the circular sweep)
    center_x = 1 / (1 + circular_delta)
    center_y = 1 / (2 * (1 + circular_delta))
    ax.plot(center_x, center_y, 'o', color='black', markersize=8, label=r'$A_{\delta} \text{ Nash Equilibrium (Center)}$')

    # Set static limits to freeze the frame
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("First Component of Player x Strategy")
    ax.set_ylabel("First Component of Player y Strategy")
    ax.legend()

def plot_duality_gap_history(results_OMWU, idx, ax):
    result = results_OMWU.list_of_results[idx]
    duality_gap_history = result.gaps_history
    ax.plot(duality_gap_history, color='purple')
    ax.set_title("Duality Gap History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Duality Gap")
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)

def plot_cumulative_total_var_history(results_OMWU, idx, ax):
    gaps_history = results_OMWU.list_of_results[idx].gaps_history

    cumulative_total_var_history = np.cumsum(np.abs(np.diff(gaps_history)))
    ax.plot(cumulative_total_var_history, color='orange')
    ax.set_title("Cumulative Total Variation")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Total Variation")
    # ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)

def iteration_wise_study(results_OMWU, idx, p_fn, q_fn, circular_delta):
    # create the canvas : 
    fig, axes = plt.subplot_mosaic([['2d_profile', 'duality_gap'],
                                   ['2d_profile', 'cumulative_total_var']],
                                  gridspec_kw={'width_ratios': [1, 1], 
                                  'height_ratios': [1, 1]}, figsize=(12, 8), constrained_layout=True)
    # plot the 2d profile :
    plot_2d_profile(results_OMWU, idx, axes['2d_profile'], p_fn, q_fn, circular_delta)

    # plot the duality gap :
    plot_duality_gap_history(results_OMWU, idx, axes['duality_gap'])

    # plot the cumulative total variation :
    plot_cumulative_total_var_history(results_OMWU, idx, axes['cumulative_total_var'])

    return fig