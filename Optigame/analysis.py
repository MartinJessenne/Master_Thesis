import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
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
def _():
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
    print(result.gaps_history)
    return (optigame,)


@app.cell
def _(np, optigame, plt):
    gammas_support = np.linspace(0, 0.25, 500)
    gammas = np.array(list(map(lambda gamma: gamma**3, gammas_support)))

    lambdas = np.array(gammas_support)

    _x = np.array([0,0], dtype=np.float64)
    _y = np.array([0,0], dtype=np.float64)

    list_of_results = []

    for (_lambda, gamma) in zip(lambdas, gammas):
        S = 1
        a = 1 + S*(1 - gamma - _lambda)
        b = 1-_lambda*S
        c = 1 - gamma*S
        d = 1
        _A = np.array([[a, b], [c, d]], dtype=np.float64)

        _g = optigame.GameState(_x, _y, _A)
        _O = optigame.Optimizer.omwuoomd(0.01, 2)

        _e = optigame.Experiment(_g, _O, 10_000)

        _result = _e.run_experiment_until_convergence_in_place()
        list_of_results.append(_result)


    # plot all gaps
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 12))

    for _result in list_of_results:
        # Only plot valid (non-zero) gaps to handle log scale correctly
        mask = _result.gaps_history > 0
        if np.any(mask):
            ax_top.plot(_result.gaps_history[mask], alpha=0.2, color='tab:blue')

    ax_top.set_yscale('log')
    ax_top.set_xlabel('Iteration') 
    ax_top.set_ylabel('Duality Gap (log)')
    ax_top.set_title('Evolution of Duality Gap for different $\lambda, \gamma$')
    ax_top.grid(True, which="both", ls="--", alpha=0.6)

    # Bottom plot: final values of x and y history vs lambda
    final_x = []
    final_y = []
    for res in list_of_results:
        # Find the last index before zero-padding
        idx = np.where(res.gaps_history > 0)[0][-1] if np.any(res.gaps_history > 0) else 0
        final_x.append(res.x_history[idx])
        final_y.append(res.y_history[idx])

    final_x = np.array(final_x)
    final_y = np.array(final_y)

    for d in range(final_x.shape[1]):
        ax_bottom.plot(lambdas, final_x[:, d], label=f'final $x_{d}$')
        ax_bottom.plot(lambdas, final_y[:, d], label=f'final $y_{d}$', linestyle='--')

    ax_bottom.set_xlabel('$\lambda$')
    ax_bottom.set_ylabel('Value')
    ax_bottom.set_title('Final Iterates vs $\lambda$')
    ax_bottom.legend()
    ax_bottom.grid(True, ls="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def load_and_plot():
    return


@app.cell
def _(load_and_plot):
    gaps, last_iterate_x, last_iterate_y = load_and_plot()
    return


if __name__ == "__main__":
    app.run()
