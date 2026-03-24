import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, np, plt


@app.cell
def load_and_plot(Path, np):
    def load_and_plot(file_name="test.npz"):

        # Locate the file dynamically
        data_path = Path(__file__).parent / file_name

        try:
            with np.load(data_path, allow_pickle=False) as data:
                gaps = data['gaps_history']
                last_iterate_x = data['last_iterate_x']
                last_iterate_y = data['last_iterate_y']

        except FileNotFoundError:
            print(f"Waiting for data file at: {data_path}")

        return gaps, last_iterate_x, last_iterate_y

    return (load_and_plot,)


@app.cell
def _(load_and_plot):
    gaps, last_iterate_x, last_iterate_y = load_and_plot()
    return


@app.cell
def _(load_and_plot, np):
    # Create a class to hold the convergence results from the experiment run
    # This class will be initialized with the gaps
    # and will compute the best iterates and random iterates for analysis

    class Convergence_Results:
        def __init__(self, historic_gaps):
            best_iterates      = np.minimum.accumulate(historic_gaps)
            cumulative_sum     = np.cumsum(historic_gaps)
            counts             = np.arange(1, len(historic_gaps) + 1)
            random_iterates    = cumulative_sum / counts

            self.last_i = historic_gaps
            self.rnd_i  = random_iterates
            self.best_i = best_iterates

    def Results_from_file(file_name="test.npz") -> Convergence_Results:
        gaps, last_iterate_x, last_iterate_y = load_and_plot(file_name)
        return Convergence_Results(gaps)

    return (Results_from_file,)


@app.cell
def _(np, plt):
    import matplotlib.ticker as ticker
    from matplotlib.ticker import LogLocator

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5,
        'figure.figsize': (6, 12),
        'axes.grid': True,
        'axes.grid.axis': 'both',
        'axes.grid.which': 'major',
        'grid.alpha': 0.6,
        'grid.linestyle': '--'
    })

    def plot_convergence_results(results_dict):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,18))
        for name, res in results_dict.items():
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

            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_minor_locator(plt.NullLocator())

            ax.grid(True, which="major", ls="-", alpha=0.6)
            ax.grid(True, which="both", linestyle="--")

        return fig

    return (plot_convergence_results,)


@app.cell
def _(Results_from_file, plot_convergence_results):
    # Load the data from your Rust experiment file
    rust_experiment_results = Results_from_file("test.npz")

    # Plot it using the imported styling exactly like in MatrixGame
    fig = plot_convergence_results({
        "OMWU (from Rust)": rust_experiment_results
    })

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
