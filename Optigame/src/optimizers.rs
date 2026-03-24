use core::f64;

use enum_dispatch::enum_dispatch;

use crate::{experiments::GameState, math::{S, V}};

pub struct Ogda {
    eta: f64,
    x_hat: S, 
    y_hat: S,
}

pub struct OmwuOomd {
    eta: f64,
    x_hat: S,
    y_hat: S, 
    grad_x_prev: V,
    grad_y_prev: V,
}

impl OmwuOomd {
    pub fn new(eta: f64, x_hat: S, y_hat: S) -> Self {
        let grad_x_prev = V::zeros(x_hat.len());
        let grad_y_prev = V::zeros(y_hat.len());
        Self {
            eta,
            x_hat,
            y_hat,
            grad_x_prev,
            grad_y_prev,
        }
    }
}

pub struct OmwuOftrl {
    eta: f64,
    grad_x_prev: V,
    grad_y_prev: V,
    cumulative_grad_x : V,
    cumulative_grad_y : V,
}

#[enum_dispatch(OptimizerStrategy)]
pub enum Optimizer {
    Ogda(Ogda),
    OmwuOomd(OmwuOomd),
    OmwuOftrl(OmwuOftrl),
}

#[enum_dispatch]
/// The OptimizerStrategy Trait implement the step method that allows, given a mutable reference to a GameState
/// instance to compute the next step. It only outputs the duality gap of the last step. 
pub trait OptimizerStrategy {
    fn step(&mut self, state: &mut GameState) -> f64;
}

impl OptimizerStrategy for Ogda {

    fn step(&mut self, state: &mut GameState) -> f64 {
        let (grad_x, grad_y) = state.compute_gradient();

        // compute steps without consuming grads
        let step_x = self.eta * &grad_x;
        let step_y = self.eta * &grad_y;

        let x_hat_next = S::from_projected(&self.x_hat - &step_x);
        let y_hat_next = S::from_projected(&self.y_hat - &step_y);

        state.x = S::from_projected(&x_hat_next - &step_x);
        state.y = S::from_projected(&y_hat_next - &step_y);

        self.x_hat = x_hat_next;
        self.y_hat = y_hat_next;

        state.duality_gap(&grad_x, &grad_y)
    }
}

impl OptimizerStrategy for OmwuOomd {
    fn step(&mut self, state: &mut GameState) -> f64 {
        let (grad_x, grad_y) = state.compute_gradient();

        // Multiplicative update of \hat{x} and \hat{y}
        let mut x_hat = self.x_hat.as_array() * (- self.eta * &grad_x).map(|&step| f64::exp(step));
        let mut y_hat = self.y_hat.as_array() * (- self.eta * &grad_y).map(|&step| f64::exp(step));

        // Normalisation de \hat{x} et \hat{y}
        x_hat /= x_hat.sum();
        y_hat /= y_hat.sum();

        // update the strategy
        let mut x = &x_hat * (- self.eta * &grad_x).map(|&step|f64::exp(step));
        let mut y = &y_hat * (- self.eta * &grad_y).map(|&step| f64::exp(step));

        x /= x.sum();
        y /= y.sum();

        // Check if they lie on the Simplex 
        state.x = S::build(x).expect("x doesn't lie on the simplex");
        state.y = S::build(y).expect("y doesn't lie on the simplex");
        
        self.x_hat = S::build(x_hat).expect("x_hat doesn't lie on the simplex");
        self.y_hat = S::build(y_hat).expect("y_hat doesn't lie on the simplex");

        state.duality_gap(&grad_x, &grad_y)
    }
}

impl OptimizerStrategy for OmwuOftrl {
    fn step(&mut self,state: &mut GameState) -> f64 {
        let (grad_x, grad_y) = state.compute_gradient();

        self.cumulative_grad_x = &self.cumulative_grad_x + &grad_x;
        self.cumulative_grad_y = &self.cumulative_grad_y + &grad_y;

        // Multiplicative update of \hat{x} and \hat{y}

        let step_x = -self.eta * (&self.cumulative_grad_x + &grad_x);
        let step_y = -self.eta * (&self.cumulative_grad_y + &grad_y);

        // update the strategy
        let mut x = state.x.as_array() * &(step_x).map(|&step| f64::exp(step));
        let mut y = state.y.as_array() * &(step_y).map(|&step| f64::exp(step));

        x /= x.sum();
        y /= y.sum();

        // Check if they lie on the Simplex 
        state.x = S::build(x).expect("x doesn't lie on the simplex");
        state.y = S::build(y).expect("y doesn't lie on the simplex");
        
        state.duality_gap(&grad_x, &grad_y)        
    }
}