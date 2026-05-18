use crate::domain::structure::GameResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    MaxNorm,
    InfinityNorm,
    Frobenius,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationMethodType {
    Concentric { num_slices: usize },
    Scattered(NormType),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricMethodType {
    MaxLast(f64),
    VarLast(f64),
    TotalVar,
}

pub struct HyperParams {
    pub num_explo: usize,
    pub num_iter_per_explo: usize,
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub metric_method: MetricMethodType,
}

pub struct Params {
    pub hyperparams: HyperParams,
    pub method: ExplorationMethodType,
}

pub struct RandomGameResult {
    pub norm: f64,
    pub game_result: GameResult,
}

pub enum ExplorationOutput {
    Concentric(ConcentricOutput),
    Scattered(ScatteredOutput),
}

pub struct ConcentricOutput {
    pub slices_boundaries: Vec<(f64, f64)>,
    pub metrics: Vec<Vec<f64>>,
}

pub struct ScatteredOutput {
    pub norms: Vec<f64>,
    pub metrics: Vec<f64>,
}
