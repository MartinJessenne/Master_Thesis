# Optigame: Last-Iterate Convergence of Learning in Games

[![CI](https://github.com/MartinJessenne/Master_Thesis/actions/workflows/CI.yml/badge.svg)](https://github.com/MartinJessenne/Master_Thesis/actions/workflows/CI.yml)
[![PyPI version](https://badge.fury.io/py/optigame.svg)](https://badge.fury.io/py/optigame)
[![Python Version](https://img.shields.io/pypi/pyversions/optigame.svg)](https://pypi.org/project/optigame/)

This repository contains the implementation, benchmarking suite, and redaction materials of my Master's Thesis. The core objective is to study and replicate the results of two papers in game theory and online learning:

1. **"Fast Last-Iterate Convergence of Learning in Games Requires Forgetful Algorithms"** (Cai et al.)
2. **"On Separation Between Best-Iterate, Random-Iterate, and Last-Iterate Convergence of Learning in Games"** (Cai et al.)

---

## 🎯 Project Aim & Scientific Context

In multi-agent reinforcement learning (MARL), the convergence of learning dynamics (like Optimistic Multiplicative Weights Update — **OMWU**) has historically been analyzed in an ergodic (time-averaged) sense. However, in non-convex landscapes (such as GAN training), time-averaging fails because in non-convex set-ups, the average of high-performing strategies often lands in regions of low performance. This necessitates a transition toward **iteration-wise (non-ergodic) convergence**.

This project investigates iteration-wise convergence modes in two-player zero-sum games, focusing on:

* **Regularizer Structure & Simplex Topology**: Investigating the impact of the regularizer choice on the learning dynamics near the simplex boundaries. We compare **OMWU** (using a negative entropy regularizer, which induces non-convergent, cyclic trajectories close to the boundaries of the strategy space) against **OGDA** (using a quadratic Euclidean regularizer, which maintains stable last-iterate convergence across the entire space).
* **Deterministic Parametric Sweeps**: Enabling deterministic sweeps along a parametric curve using a family of payoff matrices with a parameterizable Nash Equilibrium, allowing us to map the trajectory dynamics in 1D and 2D strategy spaces.
* **Trajectory-Based Chaos Metrics**: Defining three quantitative metrics to evaluate convergence and characterize oscillatory/chaotic behaviors:
  * **Maximum Value of the last X% of iterates** (measuring peak oscillation amplitude).
  * **Variance of the last X% of iterates** (quantifying strategy instability).
  * **Total Variation** (measuring cumulative step-by-step changes along the trajectory).
* **Empirical Convergence Separations**: Quantifying the mathematical separation between three non-ergodic convergence modes:
  * **Last-Iterate**: Oscillates indefinitely in a non-decaying limit cycle.
  * **Random-Iterate**: Decays extremely slowly, exhibiting a logarithmic $O(1/\log T)$ lower bound.
  * **Best-Iterate**: Converges rapidly in a staircase pattern, bounded by the theoretical $O(T^{-1/6})$ worst-case uniform envelope.

---

## ⚡ High-Performance Architecture (`optigame`)

Simulating thousands of high-iteration sweeps (e.g., $10,000$ iterations across a $500 \times 500$ parameter grid) in pure Python is computationally prohibitive due to interpreter overhead and the Global Interpreter Lock (GIL). 

To solve this, we developed **`optigame`**:
* **Rust Core**: Numerical loops, gradient computations, and simplex projections are implemented in Rust to guarantee zero-allocation performance.
* **Rayon Parallelism**: Neighborhood sweeps and random payoff matrix explorations are distributed across all CPU threads without GIL bottlenecks leveraging the rayon crate in Rust. 
* **PyO3 FFI Bridge**: Exposes a clean, idiomatic Python API. Memory is shared zero-copy using NumPy views.
* **Type Safety & IDE Support**: Fully typed using Python stubs (`.pyi` / `py.typed`) for autocompletion with the Python LSP.

---

## 📦 Installation

`optigame` is published to PyPI and can be installed via `pip`:

```bash
pip install optigame
```

For development and workspace setup (using [uv](https://github.com/astral-sh/uv)):

```bash
# Clone the repository
git clone https://github.com/MartinJessenne/Master_Thesis.git
cd Master_Thesis

# Build local workspace and compile the extension
uv sync
```

---

## 🚀 Usage & Notebooks

The experiments and interactive plots are orchestrated using [Marimo](https://github.com/marimo-team/marimo) notebooks.

To run the interactive analysis:
```bash
uv run marimo edit optigame/analysis.py
```

### Simple API Example
You can use the python library directly to run experiments:

```python
import numpy as np
import optigame

# 1. Setup a Game State (x, y, Payoff Matrix)
x = np.array([0.5, 0.5])
y = np.array([0.5, 0.5])
payoff_matrix = np.array([[0.5, 0.2], [0.1, 0.8]])
state = optigame.GameState(x, y, payoff_matrix)

# 2. Instantiate an Optimizer (e.g., OMWU via OOMD)
# eta (learning rate) = 0.1, dim = 2
optimizer = optigame.Optimizer.omwuoomd(eta=0.1, dim=2)

# 3. Run simulation
experiment = optigame.Experiment(state, optimizer, num_steps=1000)
result = experiment.run_experiment_until_convergence_in_place()

# 4. Access convergence history
print(result.gaps_history)
```

---

## 📂 Project Structure

```text
.
├── .github/workflows/    # CI/CD pipelines for cross-compiling & publishing wheels
├── optigame/             # The PyO3 Rust crate
│   ├── src/              # Rust implementation (ffi, optimizers, math)
│   ├── test/             # Integration tests
│   ├── pyproject.toml    # Maturin build configurations
│   └── optigame.pyi      # Type stubs for editor autocompletion
├── Master_Thesis.pdf     # Compiled thesis PDF
├── pyproject.toml        # Root Python workspace configurations
└── README.md             # This file
```

---

## 🎓 Citation

If you use this codebase or the accompanying thesis, please cite:

```bibtex
@mastersthesis{jessenne2026convergence,
  author       = {Martin Jessenne},
  title        = {On the Convergence Modes of Learning in Games: Best, Random, and Last-Iterate Convergence},
  school       = {HEC Paris},
  year         = {2026},
  month        = {May}
}
```
