use core::f64;
use std::f64::EPSILON;

use enum_dispatch::enum_dispatch;
use pyo3::{Python, pyclass, pymethods};

use crate::{
    domain::GameState,
    math::{S, V},
};

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
    pub fn py_new(eta: f64, dim: usize) -> Self {
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
        Self { eta, x_hat, y_hat }
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
    cumulative_grad_x: V,
    cumulative_grad_y: V,
}

impl OmwuOftrl {
    pub fn new(eta: f64, dim: usize) -> Self {
        OmwuOftrl {
            eta,
            cumulative_grad_x: V::zeros(dim),
            cumulative_grad_y: V::zeros(dim),
        }
    }
}

#[pymethods]
impl OmwuOftrl {
    #[new]
    pub fn py_new(eta: f64, dim: usize) -> Self {
        Self::new(eta, dim)
    }
}

#[enum_dispatch(OptimizerStrategy)]
#[derive(Clone)]
pub enum OptimizerEnum {
    Ogda(Ogda),
    OmwuOomd(OmwuOomd),
    OmwuOftrl(OmwuOftrl),
}

#[pyclass(name = "Optimizer")]
#[derive(Clone)]
pub struct Optimizer {
    pub inner: OptimizerEnum,
}

#[pymethods]
impl Optimizer {
    #[staticmethod]
    pub fn ogda(_py: Python<'_>, eta: f64, dim: usize) -> Self {
        let ogda = Ogda::new(eta, dim);
        Self {
            inner: OptimizerEnum::Ogda(ogda),
        }
    }

    #[staticmethod]
    pub fn omwuoomd(_py: Python<'_>, eta: f64, dim: usize) -> Self {
        let omwu = OmwuOomd::new(eta, dim);
        Self {
            inner: OptimizerEnum::OmwuOomd(omwu),
        }
    }

    #[staticmethod]
    pub fn omwuoftrl(_py: Python<'_>, eta: f64, dim: usize) -> Self {
        let omwu = OmwuOftrl::new(eta, dim);
        Self {
            inner: OptimizerEnum::OmwuOftrl(omwu),
        }
    }
}

#[enum_dispatch]
pub trait OptimizerStrategy {
    fn step(&mut self, state: &mut GameState) -> f64;

    fn reset(&mut self);
}

impl OptimizerStrategy for Optimizer {
    fn step(&mut self, state: &mut GameState) -> f64 {
        self.inner.step(state)
    }

    fn reset(&mut self) {
        self.inner.reset()
    }
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
        self.x_hat = S::build(V::from_elem(dim, 1. / (dim as f64))).unwrap();
        self.y_hat = S::build(V::from_elem(dim, 1. / (dim as f64))).unwrap()
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
        let mut x_hat = step_x.map(|&s| f64::exp(s - max_step_x));
        let mut y_hat = step_y.map(|&s| f64::exp(s - max_step_y));

        x_hat *= &self.x_hat.view();
        y_hat *= &self.y_hat.view();

        x_hat.mapv_inplace(|v| v + EPSILON);
        y_hat.mapv_inplace(|v| v + EPSILON);

        // Normalizing \hat{x} and \hat{y}
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
        self.x_hat = S::build(V::from_elem(dim, 1. / (dim as f64))).unwrap();
        self.y_hat = S::build(V::from_elem(dim, 1. / (dim as f64))).unwrap()
    }
}

impl OptimizerStrategy for OmwuOftrl {
    fn step(&mut self, state: &mut GameState) -> f64 {
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
    use crate::domain::GameState;
    use crate::math::S;
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
        assert_relative_eq!(opt_new.x_hat.view(), opt_reset.x_hat.view());
        assert_relative_eq!(opt_new.y_hat.view(), opt_reset.y_hat.view());
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
            assert_relative_eq!(state.x.view().sum(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(state.y.view().sum(), 1.0, epsilon = 1e-12);

            // Check non-negativity
            assert!(state.x.view().iter().all(|&v| v >= 0.0));
            assert!(state.y.view().iter().all(|&v| v >= 0.0));
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
