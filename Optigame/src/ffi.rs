use crate::domain::structure::{Experiment, GameResult, GameState};
use crate::math::S;
use crate::optimizers::core::Optimizer;
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
    let matrices_view = matrices.as_array();

    matrices_view
        .outer_iter()
        .into_par_iter()
        .map(|matrix_view| {
            let mut matrix = matrix_view.to_owned();

            if normalize_matrix {
                let max_component = matrix
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc: f64, &b| acc.max(b));
                let min_component = matrix.iter().fold(f64::INFINITY, |acc: f64, &b| acc.min(b));

                matrix = (matrix - min_component) / (max_component - min_component);
            }

            let x = S::build(array![0.5, 0.5]).unwrap();
            let y = S::build(array![0.5, 0.5]).unwrap();
            let mut game_state = GameState { x, y, a: matrix };

            let experiment = Experiment::new(num_steps);
            let mut optimizer = optimizer.clone();
            experiment.run_experiment_until_convergence(&mut game_state, &mut optimizer)
        })
        .collect()
}

// NOTE: py_random_neighborhood_exploration and py_random_exploration need to be updated 
// to match the new random_neighborhood_exploration signature and ExplorationOutput.
// This might require exposing more types to Python or zipping the output back into numpy arrays.
