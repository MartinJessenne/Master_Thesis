import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt


@app.cell
def _(np):
    def projection_simplex(v : np.array):
        """ projette un vecteur quelconque sur 
        le simplexe pour la norme euclidienne"""

        upper_bound : np.float = np.max(v)
        lower_bound : np.float = np.max(v) - 1 
        theta : np.float = (lower_bound + upper_bound)/2
        epsilon : np.float = 1e-5
        regression : np.array = np.maximum(0, v - theta)
        error : np.float = (np.sum(regression) - 1)

        while (error)**2 > epsilon :
            if error >= 0:
                lower_bound = theta
                theta = 0.5*(lower_bound + upper_bound)

            else :
                upper_bound = theta
                theta = 0.5*(lower_bound + upper_bound)

            regression = np.maximum(0, v - theta)
            error = (np.sum(regression) - 1)
        return regression

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
            # Pour l'optimisme, on a besoin des gradients à t-1
            # Au début (t=0), on peut supposer que g_{t-1} = g_t ou 0.
            # Ici on initialise les "prev" comme les actuels pour simplifier le pas 1.
            self.x_prev = x_init.copy()
            self.y_prev = y_init.copy()
            self.grad_x_prev = np.zeros_like(x_init)
            self.grad_y_prev = np.zeros_like(y_init)

            self.A = A
            self.eta = eta

            # Historique pour les courbes
            self.history_x = [x_init.copy()]
            self.history_y = [y_init.copy()]
            self.gaps = [duality_gap(x_init, y_init, A)]

        def _compute_gradients(self, x, y):
            # Gradient pour x (Minimizer) : A * y
            grad_x = self.A @ y
            # Gradient pour y (Maximizer) : A.T * x (on fait une montée de gradient)
            grad_y = self.A.T @ x 
            return grad_x, grad_y

        def step(self):
            raise NotImplementedError

    class OGDA(GameOptimizer):
        """Optimistic Gradient Descent Ascent"""
        def __init__(self, x_init, y_init, A, eta):
            super().__init__(x_init, y_init, A, eta)
            # Initialize hats (anchors) same as x, y
            self.x_hat = x_init.copy()
            self.y_hat = y_init.copy()

        def step(self):
            # 1. Calcul des gradients actuels
            grad_x, grad_y = self._compute_gradients(self.x, self.y)

            # 2. Mise à jour des x_hat et y_hat 
            self.x_hat = projection_simplex(self.x_hat - self.eta * grad_x)
            self.y_hat = projection_simplex(self.y_hat + self.eta * grad_y)

            # 4. Mise à jour des stratégies x et y
            # On minimise x donc on y soustrait la direction de la plus grande pente
            # On maximise y donc on y ajoute la direction de la plus grande pente
            self.x = projection_simplex(self.x_hat - self.eta * grad_x)
            self.y = projection_simplex(self.y_hat + self.eta * grad_y)

        
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


            # 3. Mise à jour Multiplicative (Exponentielle)
            # x minimisation : multiplie par exp(-eta * gradient)
            self.x = self.x * np.exp(-self.eta * grad_x)
            self.x /= np.sum(self.x) # Renormalisation (Projection KL)

            # y maximisation : multiplie par exp(+eta * gradient)
            self.y = self.y * np.exp(self.eta * grad_y)
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
            ax.grid(True, which="both", linestyle="--")

        return fig
    return (plot_convergence_results,)


@app.cell
def _(Results_OGDA_RPS, Results_OMWU_RPS, plot_convergence_results):
    plot_convergence_results({
        "OGDA RPS": Results_OGDA_RPS,
        "OMWU RPS": Results_OMWU_RPS
    })
    return


@app.cell
def _(mo):
    delta = mo.ui.slider(start=0., stop=0.5, step=1e-3, label="delta value")
    return (delta,)


@app.cell
def _(delta, mo):
    mo.hstack([delta, mo.md(f"Has value: {delta.value}")])
    return


@app.cell
def _(convergence_results, delta, experiment, np, plot_convergence_results):
    _A_delta = np.array([[1/2 + delta.value, 1/2],
                         [0, 1]])

    _x_init = np.array([1/2, 1/2])
    _y_init = np.array([1/2, 1/2])

    _eta = 0.1
    num_steps_delta : int = 4000
    ogda_delta_gaps, omwu_delta_gaps = experiment(_A_delta, _x_init, _y_init, _eta, num_steps_delta)
    Results_OGDA_Delta = convergence_results(ogda_delta_gaps)
    Results_OMWU_Delta = convergence_results(omwu_delta_gaps)
    plot_convergence_results({
        "OGDA Delta": Results_OGDA_Delta,
        "OMWU Delta": Results_OMWU_Delta
    })
    return


if __name__ == "__main__":
    app.run()
