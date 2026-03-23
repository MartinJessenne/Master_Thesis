use core::f64;

use std::fs::OpenOptions;

use ndarray::Axis;
use ndarray_npy::NpzWriter;

use crate::{experiments::InitialState, math::{M, S, V}};

pub struct OGDA;

pub struct OMWU {
    pub grad_x_prev: V,
    pub grad_y_prev: V,
}

pub struct OmwuCumulative {
    grad_x_prev: V,
    grad_y_prev: V,
    Cumulative_grad_x : V,
    Cumulative_grad_y : V,
}

pub trait OptimizerStrategy {
    fn step(&self, state: &mut InitialState) -> (S, S, f64);
}

impl OptimizerStrategy for OGDA {

    fn step(&self, state: &mut InitialState) -> (S, S, f64) {
        let (grad_x, grad_y) = state.compute_gradient();

        // compute steps without consuming grads
        let step_x = state.eta * &grad_x;
        let step_y = state.eta * &grad_y;

        let x_hat_next = S::from_projected(&state.x_hat - &step_x);
        let y_hat_next = S::from_projected(&state.y_hat - &step_y);

        state.x = S::from_projected(&x_hat_next - &step_x);
        state.y = S::from_projected(&y_hat_next - &step_y);

        state.x_hat = x_hat_next;
        state.y_hat = y_hat_next;

        (state.x.clone(), state.y.clone(), state.duality_gap(&grad_x, &grad_y))
    }
}

impl OptimizerStrategy for OMWU {
    fn step(&self, state: &mut InitialState) -> (S, S, f64) {
        let (grad_x, grad_y) = state.compute_gradient();

        // Multiplicative update of \hat{x} and \hat{y}
        let mut x_hat = state.x_hat.as_array() * (- state.eta * &grad_x).map(|&step| f64::exp(step));
        let mut y_hat = state.y_hat.as_array() * (- state.eta * &grad_y).map(|&step| f64::exp(step));

        // Normalisation de \hat{x} et \hat{y}
        x_hat /= x_hat.sum();
        y_hat /= y_hat.sum();

        // update the strategy
        let mut x = &x_hat * (- state.eta * &grad_x).map(|&step|f64::exp(step));
        let mut y = &y_hat * (- state.eta * &grad_y).map(|&step| f64::exp(step));

        x /= x.sum();
        y /= y.sum();

        // Check if they lie on the Simplex 
        state.x = S::build(x).expect("x doesn't lie on the simplex");
        state.y = S::build(y).expect("y doesn't lie on the simplex");
        
        state.x_hat = S::build(x_hat).expect("x_hat doesn't lie on the simplex");
        state.y_hat = S::build(y_hat).expect("y_hat doesn't lie on the simplex");

        (state.x.clone(), state.y.clone(), state.duality_gap(&grad_x, &grad_y))
    }
}
