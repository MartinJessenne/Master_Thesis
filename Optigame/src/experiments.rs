use std::fs::OpenOptions;
use std::vec;
use ndarray_npy::{NpzWriter};
use ndarray::Axis;
use crate::math::{V, M, S};
use crate::optimizers::{OGDA, OMWU, OmwuCumulative, OptimizerStrategy};

#[derive(Debug)]
pub struct InitialState {
    pub x: S,
    pub y: S,
    pub x_hat: S,
    pub y_hat: S,
    pub A: M,
    pub eta: f64,
}

impl InitialState {
    pub fn new(
        x: S,
        y: S,
        x_hat: S,
        y_hat: S,
        A: M,
        eta: f64,
    ) -> Self {
        InitialState { x, y, x_hat, y_hat, A, eta}
    }

    pub fn x(&self) -> &S{
        &self.x
    }

    pub fn y(&self) -> &S {
        &self.y
    }

    pub fn x_hat(&self) -> &S {
        &self.x_hat
    }

    pub fn y_hat(&self) -> &S {
        &self.y_hat
    }

    pub fn A(&self) -> &M {
        &self.A
    }

    pub fn eta(&self) -> &f64 {
        &self.eta
    }

    pub fn compute_gradient(&self) -> (V, V) {
        let grad_x: V = self.A.dot(self.y.as_array());
        let grad_y: V = -self.A.t().dot(self.x.as_array());
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

struct GameResult{
    x_history: Vec<S>,
    y_history: Vec<S>,
    gaps_history: Vec<f64>,
}

impl GameResult {
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

pub struct Multi_Experiments{
    state: InitialState,
    optimizer: Box< dyn OptimizerStrategy>,
    num_steps: usize,
}

impl Multi_Experiment {
    pub fn new(state: InitialState, optimizer: impl OptimizerStrategy, num_steps: usize) -> Self {
        let optimizer = Box::new(optimizer);
        Experiment {state, optimizer, num_steps}
    }

    /// This function is tasked with taking a InitialState initial values 
    /// And an optimizer to run the experiment for the nb_steps number of steps
    pub fn run_experiment(mut self) -> GameResult {
        let mut x_history: Vec<S> = vec![];
        let mut y_history: Vec<S> = vec![];
        let mut gaps_history: Vec<f64> = vec![];
        for _ in 0..self.num_steps {
            let (x, y, gap) = self.optimizer.step(&mut self.state);

            x_history.push(x);
            y_history.push(y);
            gaps_history.push(gap);
        }
        GameResult { x_history, y_history, gaps_history } 
    }

    pub fn run_experiment_until_convergence(self) -> GameResult {
        for _ in 0..self.num_steps {
            self.optimizer.step(&mut self.state);
            if *self.state.gaps().last().unwrap() < 1e-9 {
                return state
            };
        }
        state
    }
}