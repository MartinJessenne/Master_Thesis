use pyo3::prelude::*;

pub mod common;
pub mod domain;
pub mod experiments;
pub mod ffi;
pub mod math;
pub mod optimizers;

/// A Python module implemented in Rust.
#[pymodule]
mod optigame {
    #[pymodule_export]
    pub use crate::domain::structure::{GameResult, GameState};

    #[pymodule_export]
    pub use crate::ffi::{
        neighborhood_exploration, concentric_exploration, scattered_exploration, PyExperiment as Experiment,
        PyConcentricOutput, PyScatteredOutput,
    };

    #[pymodule_export]
    pub use crate::optimizers::core::{Ogda, OmwuOftrl, OmwuOomd, Optimizer};
}
