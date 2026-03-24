use std::fs::OpenOptions;
use ndarray_npy::{NpzWriter};
use ndarray::Axis;
use crate::math::{V, M, S};
use crate::optimizers::{Optimizer, OptimizerStrategy};

#[derive(Debug)]
pub struct GameState {
    pub x: S,
    pub y: S,
    pub a: M,
}

impl GameState {
    pub fn new(
        x: S,
        y: S,
        a: M,
    ) -> Self {
        GameState { x, y, a }
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
    pub fn duality_gap(&mut self, grad_x: &V, grad_y: &V) -> f64 {
        let max_y: f64 = -*grad_y.into_iter().max_by(|a,b| a.total_cmp(b)).unwrap();      
        let min_x: f64 = *grad_x.into_iter().min_by(|a,b| a.total_cmp(b)).unwrap();
        max_y - min_x 
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