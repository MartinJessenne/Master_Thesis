use crate::math::{M, S, V};
use crate::optimizers::core::{Optimizer, OptimizerStrategy};
use ndarray::{Array2, ArrayView1, ArrayView2};
use pyo3::pyclass;

#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    pub x: S,
    pub y: S,
    pub a: M,
}

impl GameState {
    pub fn new(x: S, y: S, a: M) -> Self {
        GameState { x, y, a }
    }

    pub fn from_matrix(a: M) -> Self {
        let (row, col) = a.dim();
        let x = S::build(V::from_elem(row, 1. / (row as f64))).expect("unable to build x");
        let y = S::build(V::from_elem(col, 1. / (col as f64))).expect("unable to build y");
        GameState { x, y, a }
    }

    pub fn reset_strats(&mut self) {
        self.x.to_uniform_inplace();
        self.y.to_uniform_inplace();
    }

    pub fn x(&self) -> &S {
        &self.x
    }

    pub fn y(&self) -> &S {
        &self.y
    }

    pub fn a<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.a.view()
    }

    pub fn compute_gradient(&self) -> (V, V) {
        let grad_x: V = self.a.dot(&self.y.view());
        let grad_y: V = -self.a.t().dot(&self.x.view());
        (grad_x, grad_y)
    }

    pub fn duality_gap(&mut self, grad_x: &V, grad_y: &V) -> f64 {
        let max_y: f64 = -grad_y.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let min_x: f64 = *grad_x.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        max_y - min_x
    }
}

#[pyclass]
pub struct GameResult {
    pub x_history: M,
    pub y_history: M,
    pub gaps_history: V,
}

impl GameResult {
    pub fn x_history<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.x_history.view()
    }

    pub fn y_history<'a>(&'a self) -> ArrayView2<'a, f64> {
        self.y_history.view()
    }

    pub fn gaps<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.gaps_history.view()
    }
}

#[derive(Clone, Debug)]
pub struct Experiment {
    pub num_steps: usize,
}

impl Experiment {
    pub fn new(num_steps: usize) -> Self {
        Experiment { num_steps }
    }

    pub fn run_experiment(&self, state: &mut GameState, optimizer: &mut Optimizer) -> GameResult {
        let num_steps = self.num_steps;
        let dim = state.x.len();
        let mut x_history = Array2::<f64>::zeros((num_steps, dim));
        let mut y_history = Array2::<f64>::zeros((num_steps, dim));

        let mut gaps_history: V = V::zeros(num_steps);
        for i in 0..self.num_steps {
            let gap = optimizer.step(state);

            x_history.row_mut(i).assign(&state.x.view());
            y_history.row_mut(i).assign(&state.y.view());
            gaps_history[i] = gap;
        }
        GameResult {
            x_history,
            y_history,
            gaps_history,
        }
    }

    pub fn run_experiment_until_convergence(
        &self,
        state: &mut GameState,
        optimizer: &mut Optimizer,
    ) -> GameResult {
        let num_steps = self.num_steps;
        let dim = state.x.len();
        let mut x_history = Array2::<f64>::zeros((num_steps, dim));
        let mut y_history = Array2::<f64>::zeros((num_steps, dim));
        let mut gaps_history: V = V::zeros(num_steps);

        let mut converged_at = num_steps;

        for i in 0..num_steps {
            // Record current state BEFORE taking the step
            x_history.row_mut(i).assign(&state.x.view());
            y_history.row_mut(i).assign(&state.y.view());

            let gap = optimizer.step(state);
            gaps_history[i] = gap;

            if gap < 10e-9 {
                converged_at = i + 1;
                break;
            }
        }

        // If we converged early, fill the rest of the history with the last recorded state
        // to avoid (0,0) points in downstream plotting.
        if converged_at < num_steps {
            let last_x = x_history.row(converged_at - 1).to_owned();
            let last_y = y_history.row(converged_at - 1).to_owned();
            for j in converged_at..num_steps {
                x_history.row_mut(j).assign(&last_x);
                y_history.row_mut(j).assign(&last_y);
                gaps_history[j] = gaps_history[converged_at - 1];
            }
        }

        GameResult {
            x_history,
            y_history,
            gaps_history,
        }
    }
}
