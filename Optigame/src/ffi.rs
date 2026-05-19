use crate::domain::structure::{Experiment, GameResult, GameState};
use crate::experiments::random_neighborhood_exploration;
use crate::experiments::types::{
    ExplorationMethodType, ExplorationOutput, HyperParams, MetricMethodType, NormType, Params,
};
use crate::math::S;
use crate::optimizers::core::Optimizer;
use ndarray::{Array2, array};
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::{Bound, IntoPy, Py, PyResult, Python, pyclass, pyfunction, pymethods};
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

#[pyclass]
pub struct PyConcentricOutput {
    #[pyo3(get)]
    pub slice_boundaries: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub metrics: Py<PyArray2<f64>>,
}

#[pyclass]
pub struct PyScatteredOutput {
    #[pyo3(get)]
    pub norms: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub metrics: Py<PyArray1<f64>>,
}

fn parse_metric_method(method_str: &str, cutoff: f64) -> PyResult<MetricMethodType> {
    match method_str.to_lowercase().as_str() {
        "max_last" | "max_last_10" => Ok(MetricMethodType::MaxLast(cutoff)),
        "var_last" | "var_last_10" => Ok(MetricMethodType::VarLast(cutoff)),
        "total_var" => Ok(MetricMethodType::TotalVar),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown metric method: {}",
            method_str
        ))),
    }
}

#[pyfunction(signature = (a_delta, optimizer, num_exploration, num_steps, inner_radius, outer_radius, num_slices, metric_method, cutoff=0.1))]
pub fn concentric_exploration<'py>(
    py: Python<'py>,
    a_delta: numpy::PyReadonlyArray2<f64>,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
    inner_radius: f64,
    outer_radius: f64,
    num_slices: usize,
    metric_method: &str,
    cutoff: f64,
) -> PyResult<PyConcentricOutput> {
    let method = parse_metric_method(metric_method, cutoff)?;
    let hyperparams = HyperParams {
        num_explo: num_exploration,
        num_iter_per_explo: num_steps,
        inner_radius,
        outer_radius,
        metric_method: method,
    };
    let params = Params {
        hyperparams,
        method: ExplorationMethodType::Concentric { num_slices },
    };

    let result = random_neighborhood_exploration(a_delta.as_array(), optimizer, params);

    if let ExplorationOutput::Concentric(out) = result {
        // Convert Vec<(f64, f64)> to Array2
        let boundaries_flat: Vec<f64> = out
            .slices_boundaries
            .into_iter()
            .flat_map(|(a, b)| vec![a, b])
            .collect();
        let boundaries_arr = Array2::from_shape_vec((num_slices, 2), boundaries_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let runs_per_slice = num_exploration / num_slices;
        let metrics_flat: Vec<f64> = out.metrics.into_iter().flatten().collect();
        let metrics_arr = Array2::from_shape_vec((num_slices, runs_per_slice), metrics_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyConcentricOutput {
            slice_boundaries: boundaries_arr.to_pyarray(py).unbind(),
            metrics: metrics_arr.to_pyarray(py).unbind(),
        })
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Expected ConcentricOutput but got ScatteredOutput",
        ))
    }
}

#[pyfunction(signature = (a_delta, optimizer, num_exploration, num_steps, inner_radius, outer_radius, norm_str, metric_method, cutoff=0.1))]
pub fn scattered_exploration<'py>(
    py: Python<'py>,
    a_delta: numpy::PyReadonlyArray2<f64>,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
    inner_radius: f64,
    outer_radius: f64,
    norm_str: &str,
    metric_method: &str,
    cutoff: f64,
) -> PyResult<PyScatteredOutput> {
    let method = parse_metric_method(metric_method, cutoff)?;

    let norm_type = match norm_str.to_lowercase().as_str() {
        "max" => NormType::MaxNorm,
        "infinity" => NormType::InfinityNorm,
        "frobenius" => NormType::Frobenius,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unknown norm type. Use 'max', 'infinity', or 'frobenius'",
            ));
        }
    };

    let hyperparams = HyperParams {
        num_explo: num_exploration,
        num_iter_per_explo: num_steps,
        inner_radius,
        outer_radius,
        metric_method: method,
    };
    let params = Params {
        hyperparams,
        method: ExplorationMethodType::Scattered(norm_type),
    };

    let result = random_neighborhood_exploration(a_delta.as_array(), optimizer, params);

    if let ExplorationOutput::Scattered(out) = result {
        Ok(PyScatteredOutput {
            norms: PyArray1::from_vec(py, out.norms).unbind(),
            metrics: PyArray1::from_vec(py, out.metrics).unbind(),
        })
    } else {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Expected ScatteredOutput but got ConcentricOutput",
        ))
    }
}
