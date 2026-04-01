
use ndarray::Axis;
use crate::math::{V, M, S};
use crate::optimizers::{Optimizer, OptimizerStrategy};
use rayon::prelude::*;
use ndarray::Array2;
use pyo3::{pymethods, Python, Py, pyclass, Bound};
use numpy::{ToPyArray, PyArray2, PyArrayMethods, PyArray1};

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

#[pyclass]
pub enum Method{
    PCA,
    Polar, 
    ElementWise,
}

#[pyclass]
pub struct Neighborhood {
    pub game_states: Vec<GameState>,
    pub pca_coordinates: Array2<f64>,
}

//#[pymethods]
//impl Neighborhood {
    //#[new]
    //pub fn py_new(
        //base_matrix_py: PyReadonlyArray2<f64>,
        //num_perturbations: usize,
        //method: Method, // Default to PCA
    //) -> Self {
        //let dims = base_matrix_py.dims();
        //let rows = dims[0];
        //let cols = dims[1];
        //let mut rng = rand::rng();
        
        //let mut game_states = Vec::<GameState>::with_capacity(num_perturbations);
        //let mut perturbations = Array3::<f64>::zeros((num_perturbations, rows, cols));

        //let pca_coordinates = Array2::<f64>::zeros((2, num_perturbations)); // At most 2 rows for plotting sake 
        //for _ in 0..num_perturbations {

            //// First let's create a perturbed initial state 
            //let perturbation = Array2::random((rows, cols), Uniform::new(-0.1, 0.1));
            
            //let a = perturbation + base_matrix_py.as_array();
            
            //let x = S::from_projected(V::zeros(rows));
            //let y = S::from_projected(V::zeros(rows));

            //game_states.push(GameState{x, y, a});

        //}

        //// Create a Linfa Dataset

    //}

//}

//pub fn run_neighborhood_exploration(
    //neighborhood: Neighborhood, 
    //optimizers: Vec<Optimizer>,
    //num_steps: usize,
//) -> (Array2<f64>, Array2<f64>) {

    //let rows = optimizers.len();
    //let cols = neighborhood.game_states.len();

    //let mut array_result = Array2::<f64>::zeros((rows, cols));
    //for (i, optimizer) in optimizers.iter().enumerate() {
        //let distances: Vec<f64> = neighborhood.game_states.par_iter().map(|state| {

            //// Run the experiment for each perturbed starting state 
            //let experiment = Experiment::new(state.clone(), optimizer.clone(), num_steps);
            //let result = experiment.run_experiment_until_convergence();
            
            //let perturbed_gaps = result.gaps();
            
            //// the retained metric is going to be the max of the duality gap in the last 10% of the iterations
            //let chaos_metric = perturbed_gaps.iter()
                                                //.enumerate()
                                                //.filter(|&(i, _)| i > (num_steps as f64 * 0.9) as usize)
                                                //.fold(f64::MIN, |acc, (_, &val)| acc.max(val));

            //chaos_metric
        //}).collect();
        //let array_row = ndarray::Array1::from_vec(distances);
        //array_result.row_mut(i).assign(&array_row);
    //}
    
    //(neighborhood.pca_coordinates, array_result)
//}
