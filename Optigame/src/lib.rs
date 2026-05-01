use pyo3::prelude::*;

pub mod domain;
pub mod experiments;
pub mod ffi;
pub mod math;
pub mod optimizers;

/// A Python module implemented in Rust.
#[pymodule]
mod optigame {
    #[pymodule_export]
    pub use crate::domain::{GameResult, GameState};

    #[pymodule_export]
    pub use crate::ffi::{
        neighborhood_exploration, py_random_exploration, py_random_neighborhood_exploration,
        PyExperiment as Experiment,
    };

    #[pymodule_export]
    pub use crate::optimizers::{Ogda, OmwuOftrl, OmwuOomd, Optimizer};
}
