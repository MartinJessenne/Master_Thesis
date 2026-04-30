use core::num;

use crate::math::{M, S, V};
use crate::optimizers::{Optimizer, OptimizerStrategy};
use ndarray::{Array, Array2, array};
use ndarray::{ArrayView2, Axis};
use ndarray_rand::rand::rngs::ThreadRng;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand_distr::{Distribution, Exp};
use ndarray_rand::{RandomExt, rand_distr, rand_distr::Uniform};
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::{Bound, Py, Python, pyclass, pyfunction, pymethods};
use rayon::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    pub x: S,
    pub y: S,
    pub a: M,
}

// Rust private methods
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

    /// Computes the duality gap for a couple of strategies (x, y)
    /// $$ Gap(x,y) = \max_{y'} (x^T A y') - \min_{x'} (x'^T A y) $$
    /// We use the already computed grad_y = -A^Tx and grad_x = Ay
    /// so \max_{y'} (x^T A y') = \max (-(-A^Tx)) = - min(grad_y)
    ///
    pub fn duality_gap(&mut self, grad_x: &V, grad_y: &V) -> f64 {
        let max_y: f64 = -*grad_y.into_iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let min_x: f64 = *grad_x.into_iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        max_y - min_x
    }
}

// Python exposed methods
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

#[pyclass]
pub struct GameResult {
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
pub struct ExperimentRunner {
    num_steps: usize,
}

impl ExperimentRunner {
    pub fn new(num_steps: usize) -> Self {
        ExperimentRunner { num_steps }
    }

    /// This function is tasked with taking a mutable reference to a GameState initial values
    /// And a mutable reference to an Optimizer to run the experiment for the nb_steps number of steps
    ///
    /// Invariants :
    /// You must ensure before calling this function that GameState has not already been run or is in a corrupted state
    /// for example by calling GameState.reset()
    /// You must also ensure that the Optimizer is in a initial state, also by calling Optimizer.reset() potentially
    pub fn run_experiment(&self, state: &mut GameState, optimizer: &mut Optimizer) -> GameResult {
        let num_steps = self.num_steps;
        let dim = state.x.len();
        let mut x_history = Array2::<f64>::zeros((num_steps, dim));
        let mut y_history = Array2::<f64>::zeros((num_steps, dim));

        let mut gaps_history: V = V::zeros(num_steps);
        for i in 0..self.num_steps {
            let gap = optimizer.step(state);

            x_history
                .push(Axis(0), state.x.as_array().view())
                .expect("Concatenation error");
            y_history
                .push(Axis(0), state.y.as_array().view())
                .expect("Concatenation error");
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

#[pymethods]
impl ExperimentRunner {
    #[new]
    pub fn py_new(num_steps: usize) -> Self {
        ExperimentRunner { num_steps }
    }
}

#[pyfunction]
pub fn neighborhood_exploration(
    p_lambda: numpy::PyReadonlyArray1<f64>,
    q_gamma: numpy::PyReadonlyArray1<f64>,
    optimizer: Optimizer,
    num_steps: usize,
    normalize_matrix: bool,
) -> Vec<GameResult> {
    // TODO: This is not optimal, there is a full matrix allocation at each step, using a buffer with map_init would be more efficient
    let list_of_results: Vec<GameResult> = p_lambda
        .as_slice()
        .expect("p_lambda must be contiguous")
        .par_iter()
        .zip(q_gamma.as_slice().expect("q_lambda must be contiguous"))
        .map(|(&lambda, &gamma)| {
            let S = 1.;
            let a = 1. + S * (1. - lambda - gamma);
            let b = 1. - gamma * S;
            let c = 1. - lambda * S;
            let d = 1.;

            let mut matrix = array![[a, b], [c, d]];

            if normalize_matrix {
                let max_component = matrix
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc: f64, &b| acc.max(b));
                let min_component = matrix.iter().fold(f64::INFINITY, |acc: f64, &b| acc.min(b));

                matrix = (matrix - min_component) / (max_component - min_component);
            }

            let x = S::from_projected(array![0., 0.]); // TODO: generalize to any dimension
            let y = S::from_projected(array![0., 0.]);

            let mut game_state = GameState {
                x: x,
                y: y,
                a: matrix,
            };
            let mut optimizer = optimizer.clone();
            let mut experiment = ExperimentRunner { num_steps };

            let result =
                experiment.run_experiment_until_convergence(&mut game_state, &mut optimizer);
            result
        })
        .collect();

    list_of_results
}

struct WorkerContext {
    state: GameState,
    opt: Optimizer,
    exp_runner: ExperimentRunner,
    rng: ThreadRng,
}

impl WorkerContext {
    fn new(a_base: ArrayView2<f64>, optimizer: Optimizer, num_steps: usize) -> Self {
        let dummy = Array2::zeros(a_base.dim());
        let state = GameState::from_matrix(dummy); // put a dummy matrix here that'll be filled in at each iteration
        let opt = optimizer.clone();
        let exp_runner = ExperimentRunner::new(num_steps);
        let rng = thread_rng();
        WorkerContext {
            state,
            opt,
            exp_runner,
            rng,
        }
    }

    fn run(&mut self, epsilon: f64, a_base: ArrayView2<f64>, dist: Uniform<f64>) -> GameResult {
        // Reset the optimizer
        self.opt.reset();

        // fill in the matrix and initial states, 0 allocation
        self.state.reset_strats();
        self.state.a.mapv_inplace(|_| dist.sample(&mut self.rng));

        // perform the perturbation
        self.state.a *= epsilon;
        self.state.a += &a_base;

        // run the experiment
        let game_result = self
            .exp_runner
            .run_experiment_until_convergence(&mut self.state, &mut self.opt);

        game_result
    }
}

pub fn random_neighborhood_exploration(
    a_base: ArrayView2<f64>,
    epsilon: f64,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
) -> Vec<GameResult> {
    let dist = Uniform::new(-1., 1.);
    // using rayon to run concurrently the iterations
    let vec_results = (1..=num_exploration)
        .into_par_iter()
        .map_init(
            || {
                let worker_context = WorkerContext::new(a_base, optimizer.clone(), num_steps);
                return worker_context;
            },
            |worker_context, _index| {
                let game_result = worker_context.run(epsilon, a_base, dist);
                return game_result;
            },
        )
        .collect();

    vec_results
}

#[pyfunction]
pub fn py_random_neighborhood_exploration(
    a_delta: numpy::PyReadonlyArray2<f64>,
    epsilon: f64,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
) -> Vec<GameResult> {
    let a_base = a_delta.as_array();

    random_neighborhood_exploration(a_base, epsilon, optimizer, num_exploration, num_steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::OmwuOomd;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_exploration_structural_integrity() {
        let num_exploration = 5;
        let num_steps = 10;
        let epsilon = 0.1;
        let a_base = array![[0.5, 0.5], [0.5, 0.5]];
        // Constructing without GIL for pure rust testing:
        let opt = Optimizer::OmwuOomd(OmwuOomd::new(0.1, 2));

        let results = random_neighborhood_exploration(
            a_base.view(),
            epsilon,
            opt,
            num_exploration,
            num_steps,
        );

        assert_eq!(results.len(), num_exploration);
        assert_eq!(results[0].gaps_history.len(), num_steps);
        assert_eq!(results[0].x_history.nrows(), num_steps);
        assert_eq!(results[0].y_history.nrows(), num_steps);
    }

    #[test]
    fn test_exploration_zero_noise_determinism() {
        let num_exploration = 10;
        let num_steps = 20;
        let epsilon = 0.0; // Zero noise means all runs should be identical
        let a_base = array![[0.8, 0.2], [0.3, 0.7]];
        let opt = Optimizer::OmwuOomd(OmwuOomd::new(0.1, 2));

        let results = random_neighborhood_exploration(
            a_base.view(),
            epsilon,
            opt,
            num_exploration,
            num_steps,
        );

        assert_eq!(results.len(), num_exploration);

        let baseline_gaps = &results[0].gaps_history;

        // If epsilon is 0, every thread should compute the exact same trajectory.
        // This proves that worker state recycling is perfectly clean.
        for res in results.iter().skip(1) {
            for (val1, val2) in baseline_gaps.iter().zip(res.gaps_history.iter()) {
                assert_relative_eq!(val1, val2, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_worker_perturbation_bounds() {
        let num_steps = 1;
        let epsilon = 0.1;
        let a_base = array![[0.5, 0.5], [0.5, 0.5]];
        let opt = Optimizer::OmwuOomd(OmwuOomd::new(0.1, 2));
        
        // 1. Instantiate the isolated WorkerContext
        let mut worker = WorkerContext::new(a_base.view(), opt, num_steps);
        let dist = Uniform::new(-1.0, 1.0);

        // 2. Execute a single run
        let _result = worker.run(epsilon, a_base.view(), dist);

        // 3. Introspect the internal state to verify the perturbation math
        let perturbed_matrix = &worker.state.a;
        
        for (original, perturbed) in a_base.iter().zip(perturbed_matrix.iter()) {
            let diff = (perturbed - original).abs();
            // Since U is in [-1.0, 1.0], epsilon * U is in [-epsilon, epsilon]
            // We add a tiny float tolerance (1e-12) to prevent precision failures
            assert!(diff <= epsilon + 1e-12, "Perturbation difference {} exceeded epsilon {}", diff, epsilon);
        }
    }
}
