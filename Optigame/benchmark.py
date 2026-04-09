import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import optigame
    from analysis import neighborhood_exploration_compute
    import time
    import pandas as pd
    import cProfile
    import pstats

    return neighborhood_exploration_compute, optigame, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance Benchmarking and Profiling
    The following cells collect execution time across all three implementations and display them in a logarithmic bar chart. The last cell runs a `cProfile` trace on the Python implementation to reveal the exact bottlenecks (e.g., Numpy allocations, pure python loops).
    """)
    return


@app.cell
def _(neighborhood_exploration_compute, optigame, plt, time):
    def benchmark_modes(points=500, steps=10_000):
        modes = ['python', 'mixed_rust', 'full_rust']
        results = {}

        optimizer = optigame.Optimizer.ogda(0.1, 2)

        # Redefine scale locally if needed to speed up the demo
        p_trans = lambda x: x**2
        q_trans = lambda x: 0.5 + x

        for mode in modes:
            print(f"Running {mode}...")
            start_time = time.perf_counter()

            # We avoid assigning the result to free up memory
            _ = neighborhood_exploration_compute(
                Optimizer=optimizer,
                Normalize_Matrix=True,
                p_transform=p_trans,
                q_transform=q_trans,
                execution_mode=mode,
                number_of_points=points
            )

            duration = time.perf_counter() - start_time
            results[mode] = duration
            print(f"Finished {mode} in {duration:.2f} seconds.")

        return results

    def plot_benchmark(times):
        fig, ax = plt.subplots(figsize=(8, 6))

        modes = list(times.keys())
        durations = list(times.values())

        bars = ax.bar(modes, durations, color=['#ff6b6b', '#ffcc5c', '#4ecdc4'])

        ax.set_yscale('log') # Important to show scale differences visually
        ax.set_ylabel('Execution Time (seconds) [Log Scale]')
        ax.set_title('Compute Efficiency: Python vs Rust Grid Optimization')
        ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')

        # Annotate bars with exact time
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                    f'{duration:.2f}s', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        return fig

    # Warning: Using the true defaults (500 pts, 10k steps) will take Python ~20 mins.
    # Change these values if you just want a quick proof of concept: e.g., (100, 1000)
    times = benchmark_modes(100, 1000)
    plot_benchmark(times)
    return


@app.cell
def _(neighborhood_exploration_compute, optigame):
    def profile_run(points=50, steps=1000, execution_mode='python'):
        """
        Run the profiler strictly on the pure Python version. We keep the grid
        and steps smaller so the profile trace doesn't take 20 minutes.
        """
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()

        _ = neighborhood_exploration_compute(
            Optimizer=optigame.Optimizer.ogda(0.1, 2),
            Normalize_Matrix=True,
            p_transform=lambda x: x**2,
            q_transform=lambda x: 0.5 + x,
            execution_mode=execution_mode,
            number_of_points=points
        )

        profiler.disable()

        # Sort by 'cumulative time' to see what the top-level bottlenecks are
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        print("--- Top 20 Time-Consuming Operations ---")
        stats.print_stats(20)

    return


if __name__ == "__main__":
    app.run()
