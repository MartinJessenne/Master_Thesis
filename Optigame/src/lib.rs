use pyo3::prelude::*;

pub mod math;
pub mod experiments;
pub mod optimizers;

/// A Python module implemented in Rust.
#[pymodule]
mod optigame {
    

    #[pymodule_export]
    pub use crate::experiments::{GameState, Experiment, neighborhood_exploration};

    #[pymodule_export]
    pub use crate::optimizers::{Ogda, OmwuOftrl, OmwuOomd, Optimizer};
}
