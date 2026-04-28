use ndarray::Axis;
use crate::math::{V, M, S};
use crate::optimizers::{Optimizer, OptimizerStrategy};
use rayon::prelude::*;
use ndarray::{Array, Array2, array};
use pyo3::{pymethods, Python, Py, pyclass, Bound, pyfunction};
use numpy::{ToPyArray, PyArray2, PyArrayMethods, PyArray1};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_rand::rand_distr::Distribution;

#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    pub x: S,
    pub y: S,
    pub a: M,
}

// Rust private methods
impl GameState {
    pub fn new(
        x : S,
        y : S,
        a : M
    ) -> Self {
        GameState {x, y , a}
    }

    pub fn x(&self) -> &S{
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

    /// Computes the duality gap for a couple of strategies (x, y)
    /// $$ Gap(x,y) = \max_{y'} (x^T A y') - \min_{x'} (x'^T A y) $$
    /// We use the already computed grad_y = -A^Tx and grad_x = Ay
    /// so \max_{y'} (x^T A y') = \max (-(-A^Tx)) = - min(grad_y)
    ///
    pub fn duality_gap(&mut self, grad_x: &V, grad_y: &V) -> f64 {
        let max_y: f64 = -*grad_y.into_iter().min_by(|a,b| a.total_cmp(b)).unwrap();      
        let min_x: f64 = *grad_x.into_iter().min_by(|a,b| a.total_cmp(b)).unwrap();
        max_y - min_x 
    }
}

// Python exposed methods
#[pymethods]
impl GameState{
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
    pub fn get_a(&self, py : Python<'_>) -> Py<PyArray2<f64>> {
        self.a.to_pyarray(py).unbind()
    }
}

#[pyclass]
pub struct GameResult{
    x_history: M,
    y_history: M,
    gaps_history: V,
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


#[pyclass]
pub struct Experiment{
    state: GameState,
    optimizer: Optimizer,
    num_steps: usize,
}

impl Experiment{
    pub fn new(state:GameState, optimizer: Optimizer, num_steps: usize) -> Self {
        Experiment { state, optimizer, num_steps }
    }

    /// This function is tasked with taking a InitialState initial values 
    /// And an optimizer to run the experiment for the nb_steps number of steps
    pub fn run_experiment(mut self) -> GameResult {
        let num_steps = self.num_steps;
        let dim = self.state.x.len();
        let mut x_history = Array2::<f64>::zeros((num_steps, dim));
        let mut y_history = Array2::<f64>::zeros((num_steps, dim));

        let mut gaps_history: V = V::zeros(num_steps);
        for i in 0..self.num_steps {
            let gap = self.optimizer.step(&mut self.state);

            x_history.push(Axis(0), self.state.x.as_array().view()).expect("Concatenation error");
            y_history.push(Axis(0), self.state.y.as_array().view()).expect("Concatenation error");
            gaps_history[i] = gap;
        }
        GameResult { x_history, y_history, gaps_history } 
    }

    pub fn run_experiment_until_convergence(mut self) -> GameResult {
        let num_steps = self.num_steps;
        let dim = self.state.x.len();
        let mut x_history = Array2::<f64>::zeros((num_steps, dim));
        let mut y_history = Array2::<f64>::zeros((num_steps, dim));

        let mut gaps_history: V = V::zeros(num_steps);
        for i in 0..self.num_steps {
            let gap = self.optimizer.step(&mut self.state);

            x_history.row_mut(i).assign(self.state.x.as_array());
            y_history.row_mut(i).assign(self.state.y.as_array());
            gaps_history[i] = gap;
            
            if gap < 10e-9 {
                return GameResult { x_history, y_history, gaps_history }
            }

        }
            GameResult { x_history, y_history, gaps_history }
    }
}

#[pymethods]
impl Experiment {
    #[new]
    pub fn py_new(
        state:GameState, 
        optimizer: Optimizer, 
        num_steps: usize,
    ) -> Self {
        Experiment {state, optimizer, num_steps}
    }

    pub fn run_experiment_until_convergence_in_place(&mut self) -> GameResult {
        let num_steps = self.num_steps;
        let dim = self.state.x.len();

        let mut x_history = Array2::<f64>::zeros((num_steps, dim));
        let mut y_history = Array2::<f64>::zeros((num_steps, dim));
        let mut gaps_history: V = V::zeros(num_steps);

        let optimizer = &mut self.optimizer;
        let state = &mut self.state;

        for i in 0..self.num_steps {
            let gap = optimizer.step(state);

            x_history.row_mut(i).assign(state.x.as_array());
            y_history.row_mut(i).assign(state.y.as_array());
            gaps_history[i] = gap;
            
            if gap < 10e-9 {
                return GameResult { x_history, y_history, gaps_history }
            }

        }
            GameResult { x_history, y_history, gaps_history }
    }
}

#[pyfunction]
pub fn neighborhood_exploration(p_lambda: numpy::PyReadonlyArray1<f64>, q_gamma: numpy::PyReadonlyArray1<f64>, optimizer: Optimizer, num_steps: usize, normalize_matrix: bool) -> Vec<GameResult> {
    let list_of_results: Vec<GameResult> = p_lambda.as_slice().expect("p_lambda must be contiguous").par_iter().zip(q_gamma.as_slice().expect("q_lambda must be contiguous")).map(|(&lambda, &gamma)| {
        let S = 1.;
        let a = 1. + S*(1. - lambda - gamma);
        let b = 1. - gamma*S;
        let c = 1. - lambda*S;
        let d = 1.;

        let mut matrix = array![[a, b], [c, d]];

        if normalize_matrix {
            let max_component = matrix.iter().fold(f64::NEG_INFINITY, |acc: f64, &b| acc.max(b));
            let min_component = matrix.iter().fold(f64::INFINITY, |acc: f64, &b| acc.min(b));
            
            matrix = (matrix - min_component) / (max_component - min_component);
        }

        let x = S::from_projected(array![0., 0.]); // TODO: generalize to any dimension
        let y = S::from_projected(array![0., 0.]);

        let game_state = GameState{x: x, y: y, a: matrix};
        let optimizer = optimizer.clone();
        let mut experiment = Experiment{state: game_state, optimizer, num_steps};

        let result = experiment.run_experiment_until_convergence_in_place();
        result
    }).collect();

    list_of_results
}

#[pyfunction]
pub fn random_neighborhood_exploration(A_delta: numpy::PyReadonlyArray2<f64>, epsilon: f64, num_exploration: usize, optimizer: Optimizer) -> V {
    let (row, col) = (A_delta.dims()[0], A_delta.dims()[1]);
    // Initialize random matrices
    let random_matrices = Array::random((num_exploration, row, col), Uniform::new(0., 10.));

    // Iterating concurrently through random_matrices to collect the results
    let vec_results = random_matrices.par_iter().map(|U| {let perturbation = A_delta + epsilon * U;
                                                          let game_result = optimizer(perturbation, )          })
}