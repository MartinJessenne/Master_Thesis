import marimo

__generated_with = "0.23.6"
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
        Path,
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
def _(np):
    # This cell is responsible for initiating different instances of parametric curves families based on relevant parameters

    def make_linear(a_x, b_x, a_y, b_y):
        """
        This function takes as input a_x, b_x, a_y, b_y,
        and returns two functions p_fn and q_fn that take as input x ranging [0, 1] and return the values of the Nash Equilibrium for point x^*_0 = a_x*x + b_x and y^*_0 = a_y*x + b_y, 
        for this specific iteration. 
        """
        def p_fn(x):
            return a_x * x + b_x

        def q_fn(x):
            return a_y * x + b_y

        return p_fn, q_fn


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


    return (make_circular,)


@app.cell
def _(make_circular, np):
    # Instantiate circular exploration 
    circular_delta = 0.1
    p_fn, q_fn = make_circular(center_x=1/(1+circular_delta), center_y=1/(2*(1+circular_delta)), r=0.5*(circular_delta/(1+circular_delta)), theta=2*np.pi)
    return circular_delta, p_fn, q_fn


@app.cell
def _(
    neighborhood_exploration_compute,
    neighborhood_exploration_plot,
    optigame,
    p_fn,
    q_fn,
):
    Ogda = optigame.Optimizer.ogda(0.1, 2)
    model_string = "OGDA"

    results_OGDA = neighborhood_exploration_compute(Ogda, p_transform = p_fn, q_transform = q_fn, number_of_points=100, num_steps=10000)
    neighborhood_exploration_plot(results_OGDA, model_string=model_string)
    return


@app.cell
def _(
    neighborhood_exploration_compute,
    neighborhood_exploration_plot,
    optigame,
    p_fn,
    q_fn,
):
    OMWU = optigame.Optimizer.omwuoomd(0.1, 2)
    results_OMWU = neighborhood_exploration_compute(OMWU, p_transform=p_fn, q_transform=q_fn, number_of_points=500, num_steps=10_000)
    neighborhood_exploration_plot(results_OMWU, model_string="OMWU", metric_type="total_var")
    return (results_OMWU,)


@app.cell
def _(mo, results_OMWU):
    iteration_idx = mo.ui.slider(0, len(results_OMWU.list_of_results) - 1, 1, value=250)
    iteration_idx
    return (iteration_idx,)


@app.cell
def _(circular_delta, iteration_idx, np, p_fn, plt, q_fn, results_OMWU):
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

    fig = iteration_wise_study(results_OMWU, iteration_idx.value, p_fn, q_fn, circular_delta)
    fig
    return (plot_2d_profile,)


@app.cell
def _(Path, plot_2d_profile, plt, results_OMWU):
    from matplotlib.pylab import f

    def batch_save_2d_profiles(results, indices, output_dir= "../images/2d_profiles", file_format="svg"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_2d_profile(results, idx, ax)
            filename = f"2d_profile_{idx:03d}.{file_format}"
            save_path = output_path / filename
            fig.savefig(save_path, format=file_format)
            plt.close(fig)  # Close the figure to free memory

    # Example usage: Save 20 profiles, step_size = len(results_OMWU.list_of_results) // 20 = 500
    step_size = len(results_OMWU.list_of_results) // 20


    SAVE_BATCH = False

    if SAVE_BATCH:
        batch_save_2d_profiles(results_OMWU, indices=range(0, len(results_OMWU.list_of_results), step_size))
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
        points_2d = np.array([x_hist[:,1], y_hist[:,1]]).T

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
    delta = 0.01
    A_delta = np.array([[1/2+ delta, 1/2],[0,1]])
    opt = optigame.Optimizer.ogda(0.1, 2)
    opt_name = "OGDA"
    metric_method = "max_last_10"

    output = optigame.concentric_exploration(
        a_delta=A_delta,
        optimizer=opt,
        num_exploration=1000,
        num_steps=10_000,
        inner_radius=0.0,
        outer_radius=0.1,
        num_slices=5,
        metric_method=metric_method,
    )

    matrix_results = output.metrics
    # We use the upper bound of each slice as the epsilon for plotting
    vec_epsilon = output.slice_boundaries[:, 1]
    return A_delta, matrix_results, metric_method, opt, opt_name, vec_epsilon


@app.cell
def _(matrix_results, metric_method, plt, ticker, vec_epsilon):
    def plot_box_and_whiskers(matrix_results, vec_epsilon):
        # 1. Create a matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        # 2. Plot the box and whiskers distribution
        ax.boxplot(matrix_results.T, positions=vec_epsilon, widths=0.005, manage_ticks=True)

        # 3. Format the plot (labels, title)
        ax.set_xlabel('Epsilon')
        ax.set_ylabel(f'{metric_method.replace("_", " ").title()}')
        ax.set_title(f'Distribution of {metric_method.replace("_", " ").title()} Across Epsilon Values')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlim(0, 0.11)
        plt.show()

        # 4. Display or return the figure
    plot_box_and_whiskers(matrix_results, vec_epsilon)
    return


@app.cell
def _(matrix_results, metric_method, np, opt_name, plt, vec_epsilon):
    # Mean plot of concentric exploration results
    def plot_mean_concentric_results(matrix_results, vec_epsilon, method_name=metric_method):
        mean_values = np.mean(matrix_results, axis=1)
        std_values = np.std(matrix_results, axis=1)
        fig = plt.figure(figsize=(10, 6))

        metric_labels = {
            "max_last_10": r"Max Duality Gap of the last 10% iterations",
            "var_last_10": r"Duality Gap Variance of the last 10% iterations",
            "total_var": r"Total Variation $\sum_t |\mathrm{Gap}^t - \mathrm{Gap}^{t-1}|$"
        }

        display_name = metric_labels.get(method_name, method_name.replace("_", " ").title())

        plt.plot(vec_epsilon, mean_values, marker='o', color='blue', label='Empirical Mean')
        plt.fill_between(
            vec_epsilon, 
            np.maximum(mean_values - std_values, 0),
            mean_values + std_values, 
            color='blue', 
            alpha=0.2, 
            label=r'$\pm 1$ Standard Deviation'
        )

        plt.xlabel(r"Perturbation Magnitude $\epsilon U$ ($L_\infty$-norm)", fontsize=14)
        plt.ylabel(display_name, fontsize=14)
        plt.title(f"Sensitivity of {opt_name} Boundary Instability\nto Neighborhood Perturbations of $A_\delta$", fontsize=15, weight='bold', pad=15)
        plt.xscale('linear')
        plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.6)
        plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='none')

        #fig.savefig(f"../images/mean_concentric_results_{method_name}_{opt_name}.svg", format="svg", bbox_inches='tight')

        plt.show()

    plot_mean_concentric_results(matrix_results, vec_epsilon, method_name=metric_method)
    return


@app.cell
def _(A_delta, opt, optigame):
    scattered_output = optigame.scattered_exploration(
        a_delta=A_delta,
        optimizer=opt,
        num_exploration=1000,
        num_steps=10_000,
        inner_radius=0.1,
        outer_radius=0.5,
        norm_str='max',
        metric_method="total_var",
        cutoff=0.1,
    )

    scattered_matrix_results = scattered_output.metrics
    scattered_vec_epsilon = scattered_output.norms
    return scattered_matrix_results, scattered_vec_epsilon


@app.cell
def _(plt, scattered_matrix_results, scattered_vec_epsilon):
    def plot_scattered_exploration_results(matrix_results, vec_epsilon):
        plt.figure(figsize=(10, 6))
        plt.scatter(vec_epsilon, matrix_results, alpha=0.5, color='blue')
        plt.xlabel('Max Norm of Initial Perturbation')
        plt.ylabel('Max Last Iteration')
        plt.title('Scattered Exploration Results')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.show()

    plot_scattered_exploration_results(scattered_matrix_results, scattered_vec_epsilon)
    return


@app.cell
def _(
    compute_l2_distances_to_ne,
    neighborhood_exploration_compute,
    optigame,
    plt,
):
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


@app.cell
def _(np, plt, results_OMWU):
    def plot_convergence_modes_comparison(results_OMWU, idx):
        gaps_history = np.array(results_OMWU.list_of_results[idx].gaps_history)
        steps = np.arange(len(gaps_history)) + 1

        last_iterate = gaps_history
        random_iterate = np.cumsum(gaps_history) / steps
        best_iterate = np.minimum.accumulate(gaps_history)

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.loglog(steps, last_iterate, label='Last-Iterate Gap', color='red', alpha=0.3)
        ax.loglog(steps, random_iterate, label='Random-Iterate (Running Avg)', color='orange', linewidth=2)
        ax.loglog(steps, best_iterate, label='Best-Iterate (Running Min)', color='green', linewidth=2.5)

        # Chose the constant C so that we can bound all iterates with the theoretical envelope
        C = np.max(best_iterate * (steps ** (1/6)))                                                                                                                 
        theoretical_bound = C * (steps ** (-1/6))                                                                                                                   
        ax.loglog(steps, theoretical_bound, label=r'Theoretical $O(T^{-1/6})$ Envelope', color='black', linestyle='--', alpha=0.8)

        ax.set_xlabel('Iteration Step $t$ (Log Scale)')
        ax.set_ylabel('Duality Gap (Log Scale)')
        ax.set_title('Separation of Convergence Modes for OMWU\nin Boundary-Adjacent Matrix Game', fontsize=14, weight='bold', pad=15)
        ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
        ax.grid(True, which="both", ls="--", alpha=0.5)

        fig.savefig(f"../images/OMWU_Convergence_Modes_Separation_step_{idx}.svg", format="svg")
        plt.show()

    # Generate the separation plot for a highly ill-conditioned run (e.g. index 250, middle of the circular sweep)
    plot_convergence_modes_comparison(results_OMWU,0)
    plot_convergence_modes_comparison(results_OMWU,125)
    plot_convergence_modes_comparison(results_OMWU,250)
    plot_convergence_modes_comparison(results_OMWU,375)
    return


if __name__ == "__main__":
    app.run()
