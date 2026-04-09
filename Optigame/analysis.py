import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from pathlib import Path

    return np, plt


@app.cell
def _(plt):
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        # Text Sizes
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,

        # Line Styles
        'lines.linewidth': 2.5,
        'figure.figsize': (6, 12),

        # Grid Configuration (New!)
        'axes.grid': True,           # Turn grid on by default
        'axes.grid.axis': 'both',    # Grid on x and y
        'axes.grid.which': 'major',  # Only draw grid for major ticks (powers of 10)
        'grid.alpha': 0.6,           # Make grid slightly transparent
        'grid.linestyle': '--'       # Dashed lines
    })
    return


@app.cell
def _(np):
    from numpy import dtype
    import optigame

    A = np.array([[0, -1, 1], 
                                                                [1, 0, -1], 
                                                                [-1, 1, 0]]
    , dtype=np.float64)
    x = np.array([5, 0, 5], dtype=np.float64)
    y = np.array([3, 4, 5], dtype=np.float64)

    G = optigame.GameState(x, y, A)

    O = optigame.Optimizer.ogda(0.1, 3)


    e = optigame.Experiment(G, O, 1000)

    result = e.run_experiment_until_convergence_in_place()
    print(type(result.x_history))
    return (optigame,)


@app.cell
def _():
    import enum
    from typing import Callable

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

    return Callable, NeighborhoodExplorationResult


@app.cell
def _(np, optigame):
    def python_neighborhood_exploration(P_lambdas, Q_gammas, Optimizer, num_steps, Normalize_Matrix):
        x = np.array([0,0], dtype=np.float64)
        y = np.array([0,0], dtype=np.float64)

        list_of_results = []
        for (p_lambda, q_gamma) in zip(P_lambdas, Q_gammas):
            # Reminder: p_lambda and q_gamma are the values of first component of the Nash Equilibrium of
            # respectively the x player and the y player.

            S = 1
            a = 1 + S*(1 - p_lambda - q_gamma)
            b = 1 - q_gamma*S 
            c = 1 - p_lambda*S
            d = 1
            matrix= np.array([[a, b], [c, d]], dtype=np.float64)

            if Normalize_Matrix:
                m = np.min(matrix)
                M = np.max(matrix)
                matrix = (matrix- m) / (M - m)

            g = optigame.GameState(x, y, matrix)
            e = optigame.Experiment(g, Optimizer, num_steps)
            result = e.run_experiment_until_convergence_in_place()
            list_of_results.append(result)

        return list_of_results

    return (python_neighborhood_exploration,)


@app.cell
def _(np):
    import sys
    import os
    sys.path.append(os.path.abspath('..'))

    # Because MatrixGame is a Marimo notebook (its top-level definitions are hidden inside cell functions),
    # we need to import it properly using mo.defs() or by dynamically executing it if we want its classes.
    # But since this file is itself a Marimo notebook, we can just use `mo.notebook()` or run it as a module 
    # if Marimo allows sharing. However, the easiest and standard workaround in Marimo to import another
    # notebook's variables is using `mo.refs()` or `import`. Wait, `import MatrixGame` technically works
    # if it is a pure Python file, but to access its cell variables, we need to extract them.

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

    return (full_python_neighborhood_exploration,)


@app.cell
def _(
    Callable,
    NeighborhoodExplorationResult,
    full_python_neighborhood_exploration,
    np,
    optigame,
    python_neighborhood_exploration,
):
    def neighborhood_exploration_compute(
        Optimizer: optigame.Optimizer,
        Normalize_Matrix: bool =True,
        p_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x**2,
        q_transform: Callable[[np.ndarray], np.ndarray] = lambda x: 0.5 + x,
        execution_mode: str = 'full_rust',
        number_of_points: int = 500
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

        x_axis= np.linspace(0, 0.25, number_of_points)


        list_of_results = []

        P_lambdas = p_transform(x_axis) # First component of theoretical NE for x player
        Q_gammas = q_transform(x_axis) # Second component of theoretical NE for y player

        # Call rust function : parallel_exploration(P_lambdas, Q_gammas, Optimizer, num_steps, Normalize_Matrix=True)
        if execution_mode == 'full_rust':
            list_of_results = optigame.neighborhood_exploration(P_lambdas, Q_gammas, Optimizer, 10_000, Normalize_Matrix)
        elif execution_mode == 'mixed_rust':
            list_of_results = python_neighborhood_exploration(P_lambdas, Q_gammas, Optimizer, 10_000, Normalize_Matrix)
        elif execution_mode == 'python':
            list_of_results = full_python_neighborhood_exploration(P_lambdas, Q_gammas, "Ogda", 10_000, Normalize_Matrix)
        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        return NeighborhoodExplorationResult(
            list_of_results=list_of_results,
            x_axis=x_axis,
            P_lambdas=P_lambdas,
            Q_gammas=Q_gammas
        )

    return


@app.cell
def _(np, plt):
    def plot_list_of_results(list_of_results, x_axis, metric_type='max_last_10'):
        fig, (ax_last_i, ax_random_i, ax_best_i) = plt.subplots(figsize=(8, 6), nrows=3, ncols=1)
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

        ax_random_i.scatter(x_axis, metrics_random_it, color='tab:blue', alpha=0.5)
        ax_last_i.scatter(x_axis, metrics_last_it, color='tab:blue', alpha=0.5)
        ax_best_i.scatter(x_axis, metrics_best_it, color='tab:blue', alpha=0.5)
        plt.show()

    return (plot_list_of_results,)


@app.cell
def _(np, plt):
    def plot_computed_vs_theoretical_NE(list_of_results, x_axis,P_lambdas, Q_gammas):
        fig, (ax_p_vs_NE, ax_q_vs_NE) = plt.subplots(2, 1, figsize=(8, 12))

        final_x = []
        final_y = []
        for res in list_of_results:
            # Find the last index before zero-padding
            idx = np.where(res.gaps_history > 0)[0][-1] if np.any(res.gaps_history > 0) else 0
            final_x.append(res.x_history[idx])
            final_y.append(res.y_history[idx])  

        final_x = np.array(final_x)
        final_y = np.array(final_y)

        ax_p_vs_NE.plot(x_axis, P_lambdas, label=f'theoretical $p$', linestyle=':')
        ax_p_vs_NE.plot(x_axis, final_x[:, 0], label=f'computed $p$')

        ax_q_vs_NE.plot(x_axis, Q_gammas, label=f'theoretical $q$', linestyle=':')
        ax_q_vs_NE.plot(x_axis, final_y[:, 0], label=f'computed $q$')

        ax_p_vs_NE.set_xlabel('$\lambda$')
        ax_p_vs_NE.set_ylabel('Value')
        ax_q_vs_NE.set_xlabel('$\gamma$')
        ax_q_vs_NE.set_ylabel('Value')
        ax_p_vs_NE.set_title('$x^*_0$ vs $\lambda$')
        ax_q_vs_NE.set_title('$y^*_0$ vs $\gamma$')
        ax_p_vs_NE.legend()
        ax_q_vs_NE.legend()
        ax_p_vs_NE.grid(True, ls="--", alpha=0.6)
        ax_q_vs_NE.grid(True, ls="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

    return (plot_computed_vs_theoretical_NE,)


@app.cell
def _(plot_computed_vs_theoretical_NE, plot_list_of_results):
    def neighborhood_exploration_plot(NeighborhoodExplorationResult: NeighborhoodExplorationResult, metric_type='max_last_10'):

        list_of_results = NeighborhoodExplorationResult.list_of_results
        x_axis= NeighborhoodExplorationResult.x_axis
        P_lambdas = NeighborhoodExplorationResult.P_lambdas
        Q_gammas = NeighborhoodExplorationResult.Q_gammas

        plot_list_of_results(list_of_results,x_axis=x_axis, metric_type=metric_type)

        plot_computed_vs_theoretical_NE(list_of_results, x_axis=x_axis, P_lambdas=P_lambdas, Q_gammas=Q_gammas)

    return


if __name__ == "__main__":
    app.run()
