import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import random
    import scipy
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    import marimo as mo
    import mpmath
    from mpmath import mp, mpf
    mp.dps = 128

    # Vectorized mpmath functions
    vec_exp = np.vectorize(mp.exp)

    def to_mp(data):
        """Converts a list or numpy array to a numpy array of mpmath.mpf objects."""
        if isinstance(data, np.ndarray):
            # Convert existing numpy array elements to mpf
            flat = [mp.mpf(x) for x in data.ravel()]
            return np.array(flat, dtype=object).reshape(data.shape)
        elif isinstance(data, list):
            # Handle list of lists for matrices
            if len(data) > 0 and isinstance(data[0], list):
                return np.array([[mp.mpf(x) for x in row] for row in data], dtype=object)
            return np.array([mp.mpf(x) for x in data], dtype=object)
        else:
            return mp.mpf(data)

    return LogLocator, mo, mp, np, plt, to_mp, vec_exp


@app.cell
def _(np):
    def projection_simplex(v : np.ndarray, z=1):
        # Solving the KKT gives 
        # w_i = max(v_i - \theta, 0) solving for the optimal \theta
        n_features: int = v.shape[0]
        u = np.sort(v)[::-1]
        # u is v  in descending order
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1] # Last index to be strictly positive
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w


    def duality_gap(x : np.ndarray, y: np.ndarray, A: np.ndarray) -> float:
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
def _(np, projection_simplex):
    vec = np.array([0.2, 0.5, 0.3, 0.8])
    simplex_proj = projection_simplex(vec)
    print("Projection of", vec, "onto the simplex:", simplex_proj)
    return


@app.cell
def _(duality_gap, mp, np, projection_simplex, to_mp, vec_exp):
    class GameOptimizer:
        def __init__(self, x_init, y_init, A, eta):
            # Ensure precision inputs
            self.x = to_mp(x_init)
            self.y = to_mp(y_init)

            self.x_hat = self.x.copy()
            self.y_hat = self.y.copy()

            self.grad_x_prev = np.zeros_like(self.x)
            self.grad_y_prev = np.zeros_like(self.y)

            self.A = to_mp(A)
            self.eta = mp.mpf(eta)

            # Historique pour les courbes
            self.history_x = [self.x.copy()]
            self.history_y = [self.y.copy()]
            self.gaps = [duality_gap(self.x, self.y, self.A)]

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
            self.x_hat = projection_simplex(self.x_hat - self.eta * grad_x)
            self.y_hat = projection_simplex(self.y_hat - self.eta * grad_y)

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
            # Utilisation de vec_exp pour le support mpmath
            self.x_hat = self.x_hat * vec_exp(-self.eta * grad_x)
            self.x_hat /= np.sum(self.x_hat)

            # y maximisation : multiplie par exp(-eta * gradient)
            self.y_hat = self.y_hat * vec_exp(-self.eta * grad_y)
            self.y_hat /= np.sum(self.y_hat)

            # 3. Mise à jour des stratégies x et y
            self.x = self.x_hat * vec_exp(-self.eta * grad_x)
            self.x /= np.sum(self.x)

            self.y = self.y_hat * vec_exp(-self.eta * grad_y)
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
def _(convergence_results, experiment, mp, to_mp):
    # Define A the rock papers scissors matrix game
    # Use explicit string initialization or integers for mpmath
    _A = to_mp([[0, -1, 1],
                [1, 0, -1],
                [-1, 1, 0]])

    _x_init = to_mp([0.5, 0.25, 0.25])
    _y_init = to_mp([0.25, 0.5, 0.25])

    _eta = mp.mpf('0.1')
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
def _(LogLocator, np, plt):
    from pylab import ndarray
    from numpy.typing import NDArray
    def plot_convergence_results(Results):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,18))
        for name, res in Results.items():
            # Convert mpmath objects to standard floats for plotting
            # Using np.array(..., dtype=float) automatically casts __float__
            last_i = np.array(res.last_i, dtype=float)
            rnd_i = np.array(res.rnd_i, dtype=float)
            best_i = np.array(res.best_i, dtype=float)

            axes[0].loglog(last_i, label=name)
            axes[1].loglog(rnd_i, label=name)
            axes[2].loglog(best_i, label=name)

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
    delta = mo.ui.slider(start=0., stop=0.5, step=1e-2, label="delta value", value=1e-2)
    num_steps_delta = mo.ui.slider(start=100, stop=1_000_000, step=100, label="number of iterations", value=5_000)
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
    mp,
    num_steps_delta,
    plot_convergence_results,
    to_mp,
):
    _A_delta = to_mp([[0.5 + delta.value, 0.5],
                      [0, 1]])

    _x_init = to_mp([0.5, 0.5])
    _y_init = to_mp([0.5, 0.5])

    _eta = mp.mpf('0.5')

    ogda_delta_gaps, omwu_delta_gaps = experiment(_A_delta, _x_init, _y_init, _eta, num_steps_delta.value)

    Results_OGDA_Delta = convergence_results(ogda_delta_gaps)
    Results_OMWU_Delta = convergence_results(omwu_delta_gaps)
    plot_convergence_results({
        "OGDA Delta Game": Results_OGDA_Delta,
        "OMWU Delta Game": Results_OMWU_Delta,
    })
    return


@app.cell
def _(
    convergence_results,
    delta,
    experiment,
    mp,
    np,
    num_steps_delta,
    plot_convergence_results,
):
    _B_delta = np.array([
        [0.5 + delta.value, 0.5 - delta.value],
        [0, 1]])

    # Initialisation recommandée (Uniforme)
    _x_init = np.array([0.5, 0.5])
    _y_init = np.array([0.5, 0.5])

    _eta = mp.mpf('1')

    ogda_B_delta_gaps, omwu_B_delta_gaps = experiment(_B_delta, _x_init, _y_init, _eta, num_steps_delta.value)

    Results_OGDA_B_Delta = convergence_results(ogda_B_delta_gaps)
    Results_OMWU_B_Delta = convergence_results(omwu_B_delta_gaps)
    plot_convergence_results({
        "OGDA B Delta Game": Results_OGDA_B_Delta,
        "OMWU B Delta Game": Results_OMWU_B_Delta,
    })
    return


@app.cell
def _(convergence_results, experiment, np, to_mp):
    def neighboorhood_exploration(Matrix, ogda_ref, omwu_ref, x_init, y_init, eta, num_steps) :
        # Soit U une matrice de perturbation uniforme de la même taille que Matrix
        # U est uniforme U(-0.5, 0.5)
        U = np.random.uniform(-0.5, 0.5, size=(Matrix.shape[0], Matrix.shape[1]))
        # On ne perturbe que la première ligne de la matrice de payoff, 
        # en effet, le paramètre delta n'affecte que la première ligne de la matrice, et on veut explorer le voisinage autour de A_0
        U[1, 0] = 0
        U[1, 1] = 0
        M_pertubed = Matrix + to_mp(U)

        ogda_gaps, omwu_gaps = experiment(M_pertubed, x_init, y_init, eta, num_steps)
        Results_OGDA_Perturbed = convergence_results(ogda_gaps)
        Results_OMWU_Perturbed = convergence_results(omwu_gaps)

        ogda_distance = np.linalg.norm(np.array(Results_OGDA_Perturbed.last_i) - ogda_ref)
        omwu_distance = np.linalg.norm(np.array(Results_OMWU_Perturbed.last_i) - omwu_ref)
        return  ogda_distance, omwu_distance, U[0, 0], U[0, 1]

    return (neighboorhood_exploration,)


@app.cell
def _(
    convergence_results,
    experiment,
    mp,
    neighboorhood_exploration,
    np,
    num_steps_delta,
    to_mp,
):
    # Matrice de référence, A_0 (delta = 0)
    _A_0 = to_mp([[0.5, 0.5],
                      [0, 1]])

    _x_init = to_mp([0.5, 0.5])
    _y_init = to_mp([0.5, 0.5])

    _eta = mp.mpf('0.5')

    # Dans experiment va calculer les trajectoires de OGDA et OMWU pour la matrice A_0, qui correspond à delta = 0.
    # et on va par la suite mesurer comment les trajectoires de OGDA et OMWU s'éloignent des trajectoires de référence. 
    ogda_ref, omwu_ref = experiment(_A_0, _x_init, _y_init, _eta, num_steps_delta.value)

    Results_OGDA = convergence_results(ogda_ref)
    Results_OMWU = convergence_results(omwu_ref)

    ogda_ref = Results_OGDA.last_i
    omwu_ref = Results_OMWU.last_i

    u_0_list = []
    u_1_list = []
    omwu_dist_list = []

    # On va faire 100 perturbations aléatoires de la matrice de payoff, et pour chacune d'entre elles, 
    # on va calculer la distance entre la trajectoire de OMWU et la trajectoire de référence,
    # On utilise les perturbations u[0, 0], u[0, 1] pour l'axe x, y pour projeter le voisinage de A_0.
    for i in range(100):
        ogda_dist, omwu_dist, u0, u1 = neighboorhood_exploration(_A_0, ogda_ref, omwu_ref, _x_init, _y_init, _eta, num_steps_delta.value)
        u_0_list.append(u0)
        u_1_list.append(u1)
        omwu_dist_list.append(omwu_dist)

    u_0_arr = np.array(u_0_list, dtype=float)
    u_1_arr = np.array(u_1_list, dtype=float)
    omwu_dist_arr = np.array(omwu_dist_list, dtype=float)

    from scipy.interpolate import griddata
    import plotly.graph_objects as go

    # Pour mieux visualiser la sensibilité de OMWU aux perturbations de la matrice de payoff, 
    # on utilise une interpolation pour créer une surface lisse à partir des points de données obtenus
    grid_u0, grid_u1 = np.mgrid[u_0_arr.min():u_0_arr.max():100j, u_1_arr.min():u_1_arr.max():100j]
    grid_z = griddata((u_0_arr, u_1_arr), omwu_dist_arr, (grid_u0, grid_u1), method='cubic')

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=grid_u0, y=grid_u1, z=grid_z,
        colorscale='viridis', opacity=0.7,
        colorbar=dict(title='OMWU Distance'),
        showscale=True,
    ))

    fig.add_trace(go.Scatter3d(
        x=u_0_arr, y=u_1_arr, z=omwu_dist_arr,
        mode='markers',
        marker=dict(size=4, color=omwu_dist_arr, colorscale='viridis', opacity=0.9),
        showlegend=False,
    ))

    fig.update_layout(
        title='Sensitivity of OMWU to Perturbations in the Payoff Matrix',
        scene=dict(
            xaxis_title='Perturbation U[0, 0]',
            yaxis_title='Perturbation U[0, 1]',
            zaxis_title='Distance to Reference OMWU Trajectory',
        ),
        width=900, height=650,
    )

    fig
    return


if __name__ == "__main__":
    app.run()
