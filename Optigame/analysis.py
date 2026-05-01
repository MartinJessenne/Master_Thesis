import marimo

__generated_with = "0.23.3"
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
        np,
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
    return (ticker,)


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
def _(iteration_idx):
    print(type(iteration_idx))
    return


@app.cell
def _(iteration_idx, plt, results_OMWU):
    def plot_2d_profile(results_OMWU, iteration_idx):
        idx = iteration_idx
        x_history = results_OMWU.list_of_results[idx].x_history
        y_history = results_OMWU.list_of_results[idx].y_history
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("2D Profile of the strategies")

        for x, y in zip(x_history, y_history):
            ax.plot(x[1], y[1], '*', color='blue', alpha=0.5)

        ax.set_xlabel("Player x Strategy")
        ax.set_ylabel("Player y Strategy")
        plt.show()
    plot_2d_profile(results_OMWU, iteration_idx.value)
    return


@app.cell
def _(results_OMWU):
    print(type(results_OMWU))
    return


@app.cell
def _(iteration_idx):
    print(type(iteration_idx))
    return


@app.cell
def _(iteration_idx, np, plt, results_OMWU):
    def compute_l2_distances_to_ne(results_OMWU, iteration_idx):
        # 1. Retrieve the specific run from the results based on the index
        theoretical_x_values = results_OMWU.P_lambdas
        theoretical_y_values = results_OMWU.Q_gammas

        run_data = results_OMWU.list_of_results[iteration_idx]

        # 2. Extract x_history and y_history
        x_hist = run_data.x_history
        y_hist = run_data.y_history

        # 3. Create a list or numpy array of the 2D points [x[1], y[1]] across all time steps
        # Note: Depending on how x_history is structured, you might need a list comprehension
        # points_2d = [ ... extract relevant indices ... ]
        points_2d = np.array([[x[1], y[1]] for x, y in zip(x_hist, y_hist)])

        # 4. Define the target Nash Equilibrium point as a numpy array
        # ne_point = ...

        ne_point = np.array([theoretical_x_values[iteration_idx], theoretical_y_values[iteration_idx]])

        # 5. Compute the differences and the L2 norm
        # distances = // use np.linalg.norm to compute the distance for each point against ne_point

        distances = np.linalg.norm(points_2d - ne_point, axis=1)

        # 6. Return the resulting sequence of distances
        # return distances
        return distances

    def plot_distance_to_ne_over_time(results_OMWU, iteration_idx):
        distances = compute_l2_distances_to_ne(results_OMWU, iteration_idx)
        plt.figure(figsize=(10, 6))
        plt.plot(distances, label='L2 Distance to NE', color='blue')
        plt.xlabel('Time Steps')
        plt.ylabel('L2 Distance')
        plt.title('Distance to Nash Equilibrium Over Time')
        plt.yscale('log')  # Logarithmic scale for better visibility
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.legend()
        plt.show()

    plot_distance_to_ne_over_time(results_OMWU, iteration_idx.value)
    return (compute_l2_distances_to_ne,)


@app.cell
def _(np, optigame):
    delta = 0.1
    A_delta = np.array([[1/2+ delta, 1/2],[0,1]])
    vec_epsilon = np.linspace(0.01, 0.1, 10)
    opt = optigame.Optimizer.omwuoomd(0.1, 2)
    matrix_results = optigame.random_exploration(A_delta, vec_epsilon, opt, num_exploration=1000, num_steps=10_000, method="max_last_10")
    return matrix_results, vec_epsilon


@app.cell
def _(matrix_results, plt, ticker, vec_epsilon):
    def plot_box_and_whiskers(matrix_results, vec_epsilon):
        # 1. Create a matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        # 2. Plot the box and whiskers distribution
        # Hint: ax.boxplot expects data where each *column* represents a distribution.
        # Your matrix_results has shape (num_epsilons, num_explorations). You must transpose it.
        # Hint 2: Use the 'positions' argument to place the boxes at the correct epsilon values on the X-axis.
        ax.boxplot(matrix_results.T, positions=vec_epsilon, widths=0.005, manage_ticks=True)

        # 3. Format the plot (labels, title)
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Max Last 10 Iterations')
        ax.set_title('Distribution of Max Last 10 Iterations Across Epsilon Values')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlim(0, 0.11)
        plt.show()

        # 4. Display or return the figure
    plot_box_and_whiskers(matrix_results, vec_epsilon)
    return


@app.cell
def _(
    compute_l2_distances_to_ne,
    neighborhood_exploration_compute,
    optigame,
    plt,
):
    # STRUCTURAL PSEUDO-CODE: Comparing Matrix Generation Methods

    def compare_methods(num_points=100, num_steps=5000):
        # 1. Define the shared optimizer
        opt = optigame.Optimizer.omwuoomd(0.1, 2)

        # 2. Run the experiment for the user setup ("A_lambda_gamma")
        res_user = neighborhood_exploration_compute(
            Optimizer=opt,
            number_of_points=num_points,
            num_steps=num_steps,
            execution_mode="full_rust",
            method="A_lambda_gamma",
        )

        # 3. Run the experiment for the literature setup ("lemma5")
        res_lemma5 = neighborhood_exploration_compute(
            Optimizer=opt,
            number_of_points=num_points,
            num_steps=num_steps,
            execution_mode="full_rust",
            method="lemma5",
        )

        # 4. Extract distances for a specific run index (e.g., index 50, middle of the simplex)
        target_idx = num_points // 2
        dist_user = compute_l2_distances_to_ne(res_user, target_idx)
        dist_lemma5 = compute_l2_distances_to_ne(res_lemma5, target_idx)

        # 5. Plot the comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dist_user, label=r"matrix $A_{\lambda,\gamma}$", color="blue")
        ax.plot(dist_lemma5, label=r"matrix Lemma 5", color="orange")
        ax.set_yscale('log')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('L2 Distance to NE')
        ax.set_title('Distance to NE Over Time for Different Matrix Generation Methods')
        ax.legend()
        plt.show()

    compare_methods()
    return


if __name__ == "__main__":
    app.run()
