use core::f64;
use std::f64::EPSILON;

use enum_dispatch::enum_dispatch;
use pyo3::{pyclass, Python, pymethods};

use crate::{experiments::GameState, math::{S, V, M}};

#[pyclass]
#[derive(Clone)]
pub struct Ogda {
    eta: f64,
    x_hat: S, 
    y_hat: S,
}

impl Ogda {
    pub fn new(eta: f64, dim: usize) -> Self {
        let x_hat = S::from_projected(V::zeros(dim));
        let y_hat = S::from_projected(V::zeros(dim));
        Ogda { eta, x_hat, y_hat }
    }
}

#[pymethods]
impl Ogda {
    #[new]
    pub fn py_new(
        eta: f64,
        dim: usize,
    ) -> Self {
        let x_hat = S::from_projected(V::zeros(dim));
        let y_hat = S::from_projected(V::zeros(dim));
        Ogda { eta, x_hat, y_hat }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct OmwuOomd {
    eta: f64,
    x_hat: S,
    y_hat: S, 
}

impl OmwuOomd {
    pub fn new(eta: f64, dim: usize) -> Self {
        let x_hat = S::from_projected(V::zeros(dim));
        let y_hat = S::from_projected(V::zeros(dim));
        Self {
            eta,
            x_hat,
            y_hat,
        }
    }
}

#[pymethods]
impl OmwuOomd {
    #[new]
    pub fn py_new(eta: f64, dim: usize) -> Self {
        Self::new(eta, dim)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct OmwuOftrl {
    eta: f64,
    cumulative_grad_x : V,
    cumulative_grad_y : V,
}

impl OmwuOftrl {
    pub fn new(eta: f64, dim: usize) -> Self {
        OmwuOftrl { eta, cumulative_grad_x: V::zeros(dim), cumulative_grad_y: V::zeros(dim) }
    }
}

#[pymethods]
impl OmwuOftrl {
    #[new]
    pub fn py_new(eta: f64, dim: usize) -> Self {
        Self::new(eta, dim)
    }
}

#[pyclass]
#[enum_dispatch(OptimizerStrategy)]
#[derive(Clone)]
pub enum Optimizer {
    Ogda(Ogda),
    OmwuOomd(OmwuOomd),
    OmwuOftrl(OmwuOftrl),
}

#[pymethods]
impl Optimizer {
    #[staticmethod]
    pub fn ogda(_py: Python<'_>, eta: f64, dim: usize) -> Self {
        // Instantiate Ogda using its existing constructor
        // and wraps it in the Optimizer::Ogda variant
        let ogda = Ogda::new(eta, dim);
        Self::Ogda(ogda)
    }

    #[staticmethod]
    pub fn omwuoomd(_py: Python<'_>, eta: f64, dim: usize) -> Self {
        // Instantiate OMWU Optimistic Online Mirror Descent using its existing constructor
        // and wraps it in the Optimizer::OmwuOomd variant
        let omwu = OmwuOomd::new(eta, dim);
        Self::OmwuOomd(omwu)
    }

    #[staticmethod]
    pub fn omwuoftrl(_py: Python<'_>, eta: f64, dim: usize) -> Self {
        // Instantiate OMWU Online Follow the Regularized Leader using its existing constructor
        // and wraps it in the Optimizer::OmwuOftrl variant
        let omwu = OmwuOftrl::new(eta, dim);
        Self::OmwuOftrl(omwu)
    }
}

#[enum_dispatch]
/// The OptimizerStrategy Trait implement the step method that allows, given a mutable reference to a GameState
/// instance to compute the next step. It only outputs the duality gap of the last step. 
pub trait OptimizerStrategy {
    fn step(&mut self, state: &mut GameState) -> f64;

    fn reset(&mut self);
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

    fn reset(&mut self) {
        let dim = self.x_hat.dim();
        self.x_hat = S::build(V::from_elem(dim, 1./(dim as f64))).unwrap();
        self.y_hat = S::build(V::from_elem(dim, 1./(dim as f64))).unwrap()
    }
}

impl OptimizerStrategy for OmwuOomd {
    fn step(&mut self, state: &mut GameState) -> f64 {
        let (grad_x, grad_y) = state.compute_gradient();

        let step_x: V = -self.eta * &grad_x;
        let step_y: V = -self.eta * &grad_y;

        let max_step_x = step_x.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
        let max_step_y = step_y.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

        // Multiplicative update of \hat{x} and \hat{y}
        let mut x_hat = self.x_hat.as_array() * step_x.map(|&s| f64::exp(s - max_step_x));
        let mut y_hat = self.y_hat.as_array() * step_y.map(|&s| f64::exp(s - max_step_y));

        x_hat.mapv_inplace(|v| v + EPSILON);
        y_hat.mapv_inplace(|v| v + EPSILON);

        // Normalisation de \hat{x} et \hat{y}
        x_hat /= x_hat.sum();
        y_hat /= y_hat.sum();

        // update the strategy
        let mut x = &x_hat * step_x.map(|&s| f64::exp(s - max_step_x));
        let mut y = &y_hat * step_y.map(|&s| f64::exp(s - max_step_y));

        x /= x.sum();
        y /= y.sum();

        // Check if they lie on the Simplex 
        state.x = S::build(x).expect("x doesn't lie on the simplex");
        state.y = S::build(y).expect("y doesn't lie on the simplex");
        
        self.x_hat = S::build(x_hat).expect("x_hat doesn't lie on the simplex");
        self.y_hat = S::build(y_hat).expect("y_hat doesn't lie on the simplex");

        state.duality_gap(&grad_x, &grad_y)
    }

    fn reset(&mut self) {
        let dim = self.x_hat.dim();
        self.x_hat = S::build(V::from_elem(dim, 1./(dim as f64))).unwrap();
        self.y_hat = S::build(V::from_elem(dim, 1./(dim as f64))).unwrap()
    }

}

impl OptimizerStrategy for OmwuOftrl {
    fn step(&mut self,state: &mut GameState) -> f64 {
        let (grad_x, grad_y) = state.compute_gradient();

        // update the cumulative gradient
        self.cumulative_grad_x = &self.cumulative_grad_x + &grad_x;
        self.cumulative_grad_y = &self.cumulative_grad_y + &grad_y;

        // Add the current gradient again, the optimism part 
        let step_x = -self.eta * (&self.cumulative_grad_x + &grad_x);
        let step_y = -self.eta * (&self.cumulative_grad_y + &grad_y);

        let max_step_x = step_x.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
        let max_step_y = step_y.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));

        // update the strategy
        let mut x = step_x.map(|&s| f64::exp(s - max_step_x));
        let mut y = step_y.map(|&s| f64::exp(s - max_step_y));

        x /= x.sum();
        y /= y.sum();

        // Check if they lie on the Simplex 
        state.x = S::build(x).expect("x doesn't lie on the simplex");
        state.y = S::build(y).expect("y doesn't lie on the simplex");
        
        state.duality_gap(&grad_x, &grad_y)        
    }

    fn reset(&mut self) {
        let dim = self.cumulative_grad_x.dim();
        self.cumulative_grad_x = V::zeros(dim);
        self.cumulative_grad_y = V::zeros(dim);
    }
    }

    #[cfg(test)]
    mod tests {
    use super::*;
    use crate::math::{S, V};
    use crate::experiments::GameState;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_ogda_reset_identity() {
        let eta = 0.1;
        let dim = 2;
        let opt_new = Ogda::new(eta, dim);
        let mut opt_reset = Ogda::new(eta, dim);

        // Pollute state (simulating a run by manually modifying hat iterates)
        opt_reset.x_hat = S::from_projected(array![10.0, -5.0]);
        opt_reset.y_hat = S::from_projected(array![-2.0, 3.0]);
        opt_reset.reset();

        // Compare internal fields to ensure reset matches a fresh constructor
        assert_relative_eq!(opt_new.x_hat.as_array(), opt_reset.x_hat.as_array());
        assert_relative_eq!(opt_new.y_hat.as_array(), opt_reset.y_hat.as_array());
    }

    #[test]
    fn test_omwu_oomd_simplex_invariant() {
        let eta = 0.1;
        let dim = 2;
        let mut opt = OmwuOomd::new(eta, dim);

        // Setup a simple bilinear game matrix
        let a = array![[0.5, 0.2], [0.1, 0.8]];
        let mut state = GameState::from_matrix(a);

        // Run multiple steps and verify the strategy always remains a valid probability distribution
        for _ in 0..100 {
            opt.step(&mut state);

            // Check normalization (sums to 1.0)
            assert_relative_eq!(state.x.as_array().sum(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(state.y.as_array().sum(), 1.0, epsilon = 1e-12);

            // Check non-negativity
            assert!(state.x.as_array().iter().all(|&v| v >= 0.0));
            assert!(state.y.as_array().iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn test_omwu_oftrl_reset_clears_memory() {
        let eta = 0.1;
        let dim = 2;
        let mut opt = OmwuOftrl::new(eta, dim);

        // Simulate accumulated history
        opt.cumulative_grad_x = array![1.0, 2.0];
        opt.cumulative_grad_y = array![-0.5, 0.5];

        opt.reset();

        // Verify that history is wiped, ensuring the next experiment starts 'forgetful'
        assert_relative_eq!(opt.cumulative_grad_x.sum(), 0.0);
        assert_relative_eq!(opt.cumulative_grad_y.sum(), 0.0);
    }
    }