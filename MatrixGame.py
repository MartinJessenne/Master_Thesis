import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    import marimo as mo
    return LogLocator, mo, np, plt


@app.cell
def _(np):
    def projection_simplex(v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w


    def duality_gap(x, y, A):
        """
        Calcule le Duality Gap pour un couple (x, y).
        Gap(x, y) = max_y' (x^T A y') - min_x' (x'^T A y)
        """
        # x^T A donne un vecteur ligne. Le max de ce vecteur est la meilleure réponse de y.
        # A y donne un vecteur colonne. Le min de ce vecteur est la meilleure réponse de x.
        val_max_y = np.max(x @ A) 
        val_min_x = np.min(A @ y)
        return val_max_y - val_min_x
    return duality_gap, projection_simplex


@app.cell
def _(duality_gap, np, projection_simplex):
    class GameOptimizer:
        def __init__(self, x_init, y_init, A, eta):
            self.x = x_init.copy()
            self.y = y_init.copy()

            self.x_hat = x_init.copy()
            self.y_hat = y_init.copy()

            self.grad_x_prev = np.zeros_like(x_init)
            self.grad_y_prev = np.zeros_like(y_init)

            self.A = A
            self.eta = eta

            # Historique pour les courbes
            self.history_x = [x_init.copy()]
            self.history_y = [y_init.copy()]
            self.gaps = [duality_gap(x_init, y_init, A)]

        def _compute_gradients(self, x, y):
            # Gradient pour x (Minimiser) : A * y
            grad_x = self.A @ y
            # Gradient pour y (Minimiser) : -A.T * x (on va faire une montée de gradient)
            grad_y = -self.A.T @ x 
            return grad_x, grad_y

        def step(self):
            raise NotImplementedError

    class OGDA(GameOptimizer):
        """Optimistic Gradient Descent Ascent"""

        def step(self):
            # 1. Calcul des gradients actuels
            grad_x, grad_y = self._compute_gradients(self.x, self.y)

            # 2. Mise à jour des x_hat et y_hat 
            self.x_hat = self.x_hat - self.eta * grad_x
            self.y_hat = self.y_hat - self.eta * grad_y

            # 4. Mise à jour des stratégies x et y
            # On minimise x donc on y soustrait la direction de la plus grande pente
            # On maximise y donc on y ajoute la direction de la plus grande pente
            self.x = projection_simplex(self.x_hat - self.eta * grad_x)
            self.y = projection_simplex(self.y_hat - self.eta * grad_y)


            """
                x_hat = prox_fn(gx_, x_hat, eta)
                x = prox_fn(gx_, x_hat, eta)
                y_hat = prox_fn(gy_, y_hat, eta)
                y = prox_fn(gy_, y_hat, eta)
            """

            # 4. Sauvegarde pour l'étape suivante
            self.grad_x_prev = grad_x
            self.grad_y_prev = grad_y

            # 5. Stockage
            self.history_x.append(self.x.copy())
            self.history_y.append(self.y.copy())
            self.gaps.append(duality_gap(self.x, self.y, self.A))


    class OMWU(GameOptimizer):
        """Optimistic Multiplicative Weights Update"""
        def step(self):
            # 1. Calcul des gradients actuels
            grad_x, grad_y = self._compute_gradients(self.x, self.y)


            # 2. Mise à jour Multiplicative de x_hat et y_hat (Exponentielle)
            # x minimisation : multiplie par exp(-eta * gradient)
            self.x_hat = self.x_hat * np.exp(-self.eta * grad_x)

            # y maximisation : multiplie par exp(-eta * gradient)
            self.y_hat = self.y_hat * np.exp(-self.eta * grad_y)

            # 3. Mise à jour des stratégies x et y
            self.x = self.x_hat * np.exp(-self.eta * grad_x)
            self.x /= np.sum(self.x)

            self.y = self.y_hat * np.exp(-self.eta * grad_y)
            self.y /= np.sum(self.y)

            # 4. Sauvegarde
            self.grad_x_prev = grad_x
            self.grad_y_prev = grad_y

            # 5. Stockage
            self.history_x.append(self.x.copy())
            self.history_y.append(self.y.copy())
            self.gaps.append(duality_gap(self.x, self.y, self.A))
    return OGDA, OMWU


@app.cell
def _(OGDA, OMWU, np):
    def experiment(A, x_init, y_init, eta, num_steps) :
        ogda_optimizer = OGDA(x_init, y_init, A, eta)
        omwu_optimizer = OMWU(x_init, y_init, A, eta)

        for _ in range(num_steps):
            ogda_optimizer.step()
            omwu_optimizer.step()

        return (ogda_optimizer.gaps, omwu_optimizer.gaps)

    # Define a struct to conveniently hold the convergence results :
    class Convergence_Results:
        def __init__(self, Last_Iterate, Random_Iterate, Best_Iterate):
            self.last_i = Last_Iterate
            self.rnd_i  = Random_Iterate
            self.best_i = Best_Iterate

    def convergence_results(historic_gaps) -> Convergence_Results:

        best_iterates      = np.minimum.accumulate(historic_gaps)
        cumulative_sum     = np.cumsum(historic_gaps)
        counts             = np.arange(1, len(historic_gaps) + 1)
        random_iterates    = cumulative_sum / counts

        return Convergence_Results(historic_gaps, random_iterates, best_iterates)
    return convergence_results, experiment


@app.cell
def _(convergence_results, experiment, np):
    # Define A the rock papers scissors matrix game
    _A = np.array([[0, -1, 1],
                  [1, 0, -1],
                  [-1, 1, 0]])

    _x_init = np.array([1/2, 1/4,1/4])
    _y_init = np.array([1/4, 1/2, 1/4])

    _eta = 0.1
    num_steps_rps : int = 4000

    ogda_rps_gaps, omwu_rps_gaps = experiment(_A, _x_init, _y_init, _eta, num_steps_rps)

    Results_OGDA_RPS = convergence_results(ogda_rps_gaps)
    Results_OMWU_RPS = convergence_results(omwu_rps_gaps)
    return Results_OGDA_RPS, Results_OMWU_RPS


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
        'figure.figsize': (10, 15),

        # Grid Configuration (New!)
        'axes.grid': True,           # Turn grid on by default
        'axes.grid.axis': 'both',    # Grid on x and y
        'axes.grid.which': 'major',  # Only draw grid for major ticks (powers of 10)
        'grid.alpha': 0.6,           # Make grid slightly transparent
        'grid.linestyle': '--'       # Dashed lines
    })
    return


@app.cell
def _(LogLocator, plt):
    def plot_convergence_results(Results):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20,40))
        for name, res in Results.items():
            axes[0].loglog(res.last_i, label=name)
            axes[1].loglog(res.rnd_i, label=name)
            axes[2].loglog(res.best_i, label=name)

        axes[0].set_title("Last Iterate Gap")
        axes[1].set_title("Random Iterate Gap")
        axes[2].set_title("Best Iterate Gap")

        for ax in axes:
            ax.legend(fontsize=15)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Duality Gap")

            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        
            # Remove minor ticks (the small lines between powers of 10)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_minor_locator(plt.NullLocator())
        
            # Enable the grid for major ticks only
            ax.grid(True, which="major", ls="-", alpha=0.6)
            ax.grid(True, which="both", linestyle="--")

        return fig
    return (plot_convergence_results,)


@app.cell
def _(Results_OGDA_RPS, Results_OMWU_RPS, plot_convergence_results):
    plot_convergence_results({
        "OGDA RPS": Results_OGDA_RPS,
        "OMWU RPS": Results_OMWU_RPS,
    })
    return


@app.cell
def _(mo):
    delta = mo.ui.slider(start=0., stop=0.5, step=1e-3, label="delta value", value=1e-2)
    num_steps_delta = mo.ui.slider(start=100, stop=1_000_000, step=100, label="number of iterations", value=10_000)
    return delta, num_steps_delta


@app.cell
def _(delta, mo):
    mo.hstack([delta, mo.md(f"Has value: {delta.value}")])
    return


@app.cell
def _(mo, num_steps_delta):
    mo.hstack([num_steps_delta, mo.md(f"Has value: {num_steps_delta.value}")])
    return


@app.cell
def _(
    convergence_results,
    delta,
    experiment,
    np,
    num_steps_delta,
    plot_convergence_results,
):
    _A_delta = np.array([[1/2 + delta.value, 1/2],
                         [0, 1]])

    _x_init = np.array([1/2, 1/2])
    _y_init = np.array([1/2, 1/2])

    _eta = 0.1

    ogda_delta_gaps, omwu_delta_gaps = experiment(_A_delta, _x_init, _y_init, _eta, num_steps_delta.value)

    Results_OGDA_Delta = convergence_results(ogda_delta_gaps)
    Results_OMWU_Delta = convergence_results(omwu_delta_gaps)
    plot_convergence_results({
        "OGDA Delta Game": Results_OGDA_Delta,
        "OMWU Delta Game": Results_OMWU_Delta,
    })
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
