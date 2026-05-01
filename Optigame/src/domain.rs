use crate::math::{M, S, V};
use crate::optimizers::{Optimizer, OptimizerStrategy};
use ndarray::Array2;
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

    pub fn a(&self) -> &M {
        &self.a
    }

    pub fn compute_gradient(&self) -> (V, V) {
        let grad_x: V = self.a.dot(self.y.as_array());
        let grad_y: V = -self.a.t().dot(self.x.as_array());
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
    pub fn x_history(&self) -> &M {
        &self.x_history
    }

    pub fn y_history(&self) -> &M {
        &self.y_history
    }

    pub fn gaps(&self) -> &V {
        &self.gaps_history
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

            x_history.row_mut(i).assign(state.x.as_array());
            y_history.row_mut(i).assign(state.y.as_array());
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
        for i in 0..self.num_steps {
            let gap = optimizer.step(state);

            x_history.row_mut(i).assign(state.x.as_array());
            y_history.row_mut(i).assign(state.y.as_array());
            gaps_history[i] = gap;

            if gap < 10e-9 {
                return GameResult {
                    x_history,
                    y_history,
                    gaps_history,
                };
            }
        }
        GameResult {
            x_history,
            y_history,
            gaps_history,
        }
    }
}
