use crate::domain::{Experiment, GameResult, GameState};
use crate::experiments::{self, random_exploration};
use crate::math::S;
use crate::optimizers::Optimizer;
use ndarray::array;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::{Bound, Py, Python, pyclass, pyfunction, pymethods};
use rayon::prelude::*;
use std::ops::DerefMut;

#[pymethods]
impl GameState {
    #[new]
    pub fn py_new(
        x: numpy::PyReadonlyArray1<f64>,
        y: numpy::PyReadonlyArray1<f64>,
        a: numpy::PyReadonlyArray2<f64>,
    ) -> Self {
        let x = S::from_projected(x.to_owned_array());
        let y = S::from_projected(y.to_owned_array());
        let a = a.to_owned_array();
        GameState { x, y, a }
    }

    #[getter]
    pub fn get_a(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.a.to_pyarray(py).unbind()
    }
}

#[pymethods]
impl GameResult {
    #[getter(x_history)]
    pub fn py_x_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.x_history.to_pyarray(py)
    }

    #[getter(y_history)]
    pub fn py_y_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.y_history.to_pyarray(py)
    }

    #[getter(gaps_history)]
    pub fn py_gaps_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.gaps_history.to_pyarray(py)
    }
}

#[pyclass(name = "Experiment")]
pub struct PyExperiment {
    pub num_steps: usize,
    pub state: Py<GameState>,
    pub optimizer: Py<Optimizer>,
}

#[pymethods]
impl PyExperiment {
    #[new]
    pub fn py_new(state: Py<GameState>, optimizer: Py<Optimizer>, num_steps: usize) -> Self {
        PyExperiment {
            num_steps,
            state,
            optimizer,
        }
    }

    pub fn run_experiment_until_convergence_in_place(&self, py: Python<'_>) -> GameResult {
        let mut state = self.state.bind(py).borrow_mut();
        let mut opt = self.optimizer.bind(py).borrow_mut();

        let experiment = Experiment::new(self.num_steps);
        experiment.run_experiment_until_convergence(state.deref_mut(), opt.deref_mut())
    }
}

#[pyfunction]
pub fn neighborhood_exploration(
    matrices: numpy::PyReadonlyArray3<f64>,
    optimizer: Optimizer,
    num_steps: usize,
    normalize_matrix: bool,
) -> Vec<GameResult> {
    // TODO: Refactor body to iterate over the 3D matrices array
    // 1. Convert the PyReadonlyArray3 to a Rust ArrayView3
    let matrices_view = matrices.as_array();

    // 2. Iterate over the matrices in parallel.
    // API Note: `.outer_iter()` yields a sequence of 2D views (ArrayView2),
    // which represent your individual 2x2 matrices.
    matrices_view
        .outer_iter()
        .into_par_iter()
        .map(|matrix_view| {
            // 3. Convert the 2D view into an owned Array2 so you can mutate it
            let mut matrix = matrix_view.to_owned();

            // 4. Implement the normalization logic here (same as before)
            if normalize_matrix {
                let max_component = matrix
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc: f64, &b| acc.max(b));
                let min_component = matrix.iter().fold(f64::INFINITY, |acc: f64, &b| acc.min(b));

                matrix = (matrix - min_component) / (max_component - min_component);
            }

            // 5. Initialize GameState and Experiment, and run
            let x = S::build(array![0.5, 0.5]).unwrap();
            let y = S::build(array![0.5, 0.5]).unwrap();
            let mut game_state = GameState { x, y, a: matrix };
            // ...

            let experiment = Experiment::new(num_steps);
            let mut optimizer = optimizer.clone();
            experiment.run_experiment_until_convergence(&mut game_state, &mut optimizer)
        })
        .collect()
}

#[pyfunction(name = "random_neighborhood_exploration")]
pub fn py_random_neighborhood_exploration(
    a_delta: numpy::PyReadonlyArray2<f64>,
    epsilon: f64,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
) -> Vec<GameResult> {
    let a_base = a_delta.as_array();

    experiments::random_neighborhood_exploration(
        a_base,
        epsilon,
        optimizer,
        num_exploration,
        num_steps,
    )
}

#[pyfunction(name = "random_exploration")]
pub fn py_random_exploration<'py>(
    py: Python<'py>,
    a_delta: numpy::PyReadonlyArray2<f64>,
    vec_epsilon: numpy::PyReadonlyArray1<f64>,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
    method: &str,
) -> Bound<'py, numpy::PyArray2<f64>> {
    let vec_epsilon = vec_epsilon.as_array();
    let a_delta = a_delta.as_array();

    let result = random_exploration(
        a_delta,
        vec_epsilon,
        optimizer,
        num_exploration,
        num_steps,
        method,
    );

    result.to_pyarray(py)
}
