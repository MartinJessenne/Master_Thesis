import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from pathlib import Path
    from analysis_utils import neighborhood_exploration_compute, neighborhood_exploration_plot
    import optigame

    return (
        mo,
        neighborhood_exploration_compute,
        neighborhood_exploration_plot,
        optigame,
        plt,
    )


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

        'axes.grid': True,           # Turn grid on by default
        'axes.grid.axis': 'both',    # Grid on x and y
        'axes.grid.which': 'major',  # Only draw grid for major ticks (powers of 10)
        'grid.alpha': 0.6,           # Make grid slightly transparent
        'grid.linestyle': '--'       # Dashed lines
    })
    return


@app.cell
def _(
    neighborhood_exploration_compute,
    neighborhood_exploration_plot,
    optigame,
):
    Ogda = optigame.Optimizer.ogda(0.1, 2)
    results_OGDA = neighborhood_exploration_compute(Ogda)
    neighborhood_exploration_plot(results_OGDA)
    return


@app.cell
def _(
    neighborhood_exploration_compute,
    neighborhood_exploration_plot,
    optigame,
):
    OMWU = optigame.Optimizer.omwuoomd(0.1, 2)
    results_OMWU = neighborhood_exploration_compute(OMWU, number_of_points=500, num_steps=10_000)
    neighborhood_exploration_plot(results_OMWU)
    return (results_OMWU,)


@app.cell
def _(mo, results_OMWU):
    iteration_idx = mo.ui.slider(0, len(results_OMWU.list_of_results),1)
    iteration_idx
    return (iteration_idx,)


@app.cell
def _(iteration_idx, plt, results_OMWU):
    idx = iteration_idx.value
    x_history = results_OMWU.list_of_results[idx].x_history
    y_history = results_OMWU.list_of_results[idx].y_history
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("2D Profile of the strategies")

    for x, y in zip(x_history, y_history):
        ax.plot(x[1], y[1], '*', color='blue', alpha=0.5)

    ax.set_xlabel("Player x Strategy")
    ax.set_ylabel("Player y Strategy")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
