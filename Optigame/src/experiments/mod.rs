pub mod engine;
pub mod maths;
pub mod metrics;
pub mod orchestrator;
pub mod types;

use crate::optimizers::core::Optimizer;
use ndarray::ArrayView2;
use types::{ConcentricOutput, ExplorationMethodType, ExplorationOutput, Params, ScatteredOutput};

/// This function is the higher level API, this is what get exposed and how you run the explorations
pub fn random_neighborhood_exploration(
    matrix: ArrayView2<f64>,
    optimizer: Optimizer,
    params: Params,
) -> ExplorationOutput {
    match params.method {
        ExplorationMethodType::Concentric { num_slices } => ExplorationOutput::Concentric(
            ConcentricOutput::new(matrix, optimizer, params.hyperparams, num_slices),
        ),
        ExplorationMethodType::Scattered(norm) => ExplorationOutput::Scattered(
            ScatteredOutput::new(matrix, optimizer, params.hyperparams, norm),
        ),
    }
}
