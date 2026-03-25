use std::fs::OpenOptions;
use ndarray_npy::{NpzWriter};
use ndarray::{Array, Axis, Dim, array};
use crate::math::{V, M, S};
use crate::optimizers::{OmwuOftrl, Optimizer, OptimizerStrategy};
use rayon::prelude::*;
use rand::{Rng, RngExt};
use linfa::traits::{Fit, Predict};
use linfa::Dataset;
use linfa_reduction::Pca;
use ndarray::Array2;

#[derive(Debug)]
#[pyclass]
pub struct GameState {
    pub x: S,
    pub y: S,
    pub a: M,
}

// Rust private methods
impl GameState {
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
    pub fn duality_gap(&mut self, grad_x: &V, grad_y: &V) -> f64 {
        let max_y: f64 = -*grad_y.into_iter().max_by(|a,b| a.total_cmp(b)).unwrap();      
        let min_x: f64 = *grad_x.into_iter().min_by(|a,b| a.total_cmp(b)).unwrap();
        max_y - min_x 
    }
}

// Python exposed methods
#[pymethods]
impl GameState{
    #[new]
    pub fn new(
        x: V,
        y: V,
        a: M,
    ) -> Self {
        GameState { S::from_projected(x), S::from_projected(y), a }
    }
}

pub struct GameResult{
    x_history: Vec<S>,
    y_history: Vec<S>,
    gaps_history: Vec<f64>,
}

impl GameResult {
    pub fn x_history(&self) -> &Vec<S> {
        &self.x_history
    }

    pub fn y_history(&self) -> &Vec<S> {
        &self.y_history
    }

    pub fn gaps(&self) -> &Vec<f64> {
        &self.gaps_history
    }

    pub fn save_to_npz(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Automatically create the directory tree if it doesn't exist
        if let Some(parent) = std::path::Path::new(filename).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new().read(true).write(true).create(true).open(filename)?;
        let mut npz = NpzWriter::new(file);
        
        let gaps_history = ndarray::Array1::from_vec(self.gaps_history.clone());
        npz.add_array("gaps_history", &gaps_history)?;

        let last_i_x_views: Vec<_> = self.x_history.iter().map(|s| s.as_array().view()).collect();
        let last_i_x_matrix= ndarray::stack(Axis(0),&last_i_x_views)?;
        npz.add_array("last_iterate_x", &last_i_x_matrix)?;

        let last_i_y_views: Vec<_> = self.y_history.iter().map(|s| s.as_array().view()).collect();
        let last_i_y_matrix = ndarray::stack(Axis(0),&last_i_y_views)?;
        npz.add_array("last_iterate_y", &last_i_y_matrix)?;

        npz.finish()?;

        Ok(())
    }
    
}

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
        let mut x_history: Vec<S> = vec![];
        let mut y_history: Vec<S> = vec![];
        let mut gaps_history: Vec<f64> = vec![];
        for _ in 0..self.num_steps {
            let gap = self.optimizer.step(&mut self.state);

            x_history.push(self.state.x.clone());
            y_history.push(self.state.y.clone());
            gaps_history.push(gap);
        }
        GameResult { x_history, y_history, gaps_history } 
    }

    pub fn run_experiment_until_convergence(mut self) -> GameResult {
        let mut x_history: Vec<S> = vec![];
        let mut y_history: Vec<S> = vec![];
        let mut gaps_history: Vec<f64> = vec![];
        for _ in 0..self.num_steps {
            let gap = self.optimizer.step(&mut self.state);

            x_history.push(self.state.x.clone());
            y_history.push(self.state.y.clone());
            gaps_history.push(gap);
            
            if gap < 10e-9 {
                return GameResult { x_history, y_history, gaps_history }
            }

        }
            GameResult { x_history, y_history, gaps_history }
    }
}

pub struct Neighborhood {
    pub game_states: Vec<GameState>,
    pub pca_coordinates: Array2<f64>,
}

impl Neighborhood {
    pub fn new(base_matrix: &M, num_perturbations: usize) -> Self {
        let mut rng = rand::rng();
        
        let mut game_states = Vec::with_capacity(num_perturbations);
        let mut perturbations_flat = Vec::with_capacity(num_perturbations * 4);

        for _ in 0..num_perturbations {
            // Generate full 2x2 random noise
            let u00: f64 = rng.random_range(-0.1..0.1);
            let u01: f64 = rng.random_range(-0.1..0.1);
            let u10: f64 = rng.random_range(-0.1..0.1);
            let u11: f64 = rng.random_range(-0.1..0.1);

            perturbations_flat.push(u00);
            perturbations_flat.push(u01);
            perturbations_flat.push(u10);
            perturbations_flat.push(u11);

            let perturbation = ndarray::array![
                [u00, u01],
                [u10, u11]
            ];
            
            let a = base_matrix + &perturbation;
            
            let x = S::from_projected(array![0.5, 0.5]);
            let y = S::from_projected(array![0.5, 0.5]);
            
            game_states.push(GameState::new(x, y, a));
        }

        // Convert the flat Vec into an N x 4 ndarray matrix
        let perturbation_matrix = Array2::from_shape_vec((num_perturbations, 4), perturbations_flat)
            .expect("Failed to reshape perturbations");

        // Create a Linfa Dataset
        let dataset = Dataset::from(perturbation_matrix);

        // Fit the PCA model to extract the Top 2 Principal Components
        let pca_model = Pca::params(2)
            .fit(&dataset)
            .expect("Failed to compute PCA");

        // Transform the 4D data down to 2D
        let pca_coordinates = pca_model.predict(&dataset);

        Self {
            game_states,
            pca_coordinates,
        }
    }
}

pub fn run_neighborhood_exploration(
    neighborhood: Neighborhood, 
    optimizers: Vec<Optimizer>,
    num_steps: usize,
) -> (Array2<f64>, Vec<f64>) {

    let rows = optimizers.len();
    let cols = num_steps;

    let array_result: Array<f64, Dim((rows, cols))> = Array::reserve(&mut self, axis, additional); // A hashmap might be better suited ?. 
    for (i, optimizer) in optimizers.iter().enumerate() {
    let distances: Vec<f64> = neighborhood.game_states.into_par_iter().map(|state| {

        // Run the experiment for each perturbed starting state 
        let experiment = Experiment::new(state, optimizer, num_steps);
        let result = experiment.run_experiment_until_convergence();
        
        let perturbed_gaps = result.gaps();
        
        // the retained metric is going to be the max of the duality gap in the last 10% of the iterations
        let chaos_metric = perturbed_gaps.iter()
                                              .enumerate()
                                              .filter(|&(i, val)| i > (num_steps as f64 * 0.9) as usize)
                                              .fold(f64::MIN, |acc, (_, &val)| acc.max(val));

        chaos_metric
    }).collect();
    array_result.push(distances);
    }
    
    (neighborhood.pca_coordinates, distances)
}

pub fn save_sweep_to_npz(filename: &str, pca_coords: &Array2<f64>, distances: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    // Automatically create the directory tree if it doesn't exist
    if let Some(parent) = std::path::Path::new(filename).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(filename)?;
    let mut npz = NpzWriter::new(file);
    
    npz.add_array("pca_coords", pca_coords)?;
    
    let dist_array = ndarray::Array1::from_vec(distances.to_vec());
    npz.add_array("distances", &dist_array)?;
    
    npz.finish()?;
    Ok(())
}