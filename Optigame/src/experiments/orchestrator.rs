use crate::domain::structure::GameResult;
use super::engine::{Exploration, run_batch_exploration};
use crate::experiments::maths::{frobenius_norm, infinity_norm, max_norm};
use super::metrics::vec_results_to_metrics;
use super::types::{ConcentricOutput, HyperParams, NormType, RandomGameResult, ScatteredOutput};
use crate::optimizers::core::Optimizer;
use ndarray::ArrayView2;
use rand::distr::Uniform;
use rayon::prelude::*;

impl ConcentricOutput {
    pub(crate) fn new(
        matrix: ArrayView2<f64>,
        optimizer: Optimizer,
        hyperparams: HyperParams,
        num_slices: usize,
    ) -> Self {
        let mut slices_boundaries = Vec::with_capacity(num_slices);
        let mut metrics = Vec::new();

        let explorations_per_slice = hyperparams.num_explo / num_slices;
        let rayon = hyperparams.outer_radius - hyperparams.inner_radius;
        let step_size = rayon / (num_slices as f64);

        for idx in 0..num_slices {
            let lower_bound = hyperparams.inner_radius + (idx as f64) * step_size;
            let higher_bound = hyperparams.inner_radius + (idx as f64 + 1.0) * step_size;
            slices_boundaries.push((lower_bound, higher_bound));

            let unif = Uniform::new(lower_bound, higher_bound).unwrap();
            let dummy_norm_fn = max_norm;

            let vec_random_game_results = run_batch_exploration(
                matrix,
                optimizer.clone(),
                unif,
                explorations_per_slice,
                hyperparams.num_iter_per_explo,
                dummy_norm_fn,
            );

            let vec_game_results: Vec<GameResult> = vec_random_game_results
                .into_iter()
                .map(|r| r.game_result)
                .collect();
            
            metrics.push(vec_results_to_metrics(vec_game_results, hyperparams.metric_method));
        }

        ConcentricOutput {
            slices_boundaries,
            metrics,
        }
    }
}

impl ScatteredOutput {
    pub(crate) fn new(
        matrix: ArrayView2<f64>,
        optimizer: Optimizer,
        hyperparams: HyperParams,
        norm_type: NormType,
    ) -> Self {
        let norm_fn = match norm_type {
            NormType::MaxNorm => max_norm,
            NormType::InfinityNorm => infinity_norm,
            NormType::Frobenius => frobenius_norm,
        };

        let unif = Uniform::new(hyperparams.inner_radius, hyperparams.outer_radius).unwrap();

        let vec_random_game_results = (0..hyperparams.num_explo)
            .into_par_iter()
            .map_init(
                || Exploration::context(matrix, optimizer.clone(), hyperparams.num_iter_per_explo, unif, norm_fn),
                |worker, _| worker.execute_clean_run(),
            )
            .collect::<Vec<RandomGameResult>>();

        let (vec_game_results, norms): (Vec<GameResult>, Vec<f64>) = vec_random_game_results
            .into_iter()
            .map(|r| (r.game_result, r.norm))
            .unzip();

        let metrics = vec_results_to_metrics(vec_game_results, hyperparams.metric_method);

        ScatteredOutput { norms, metrics }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::types::{MetricMethodType};
    use crate::optimizers::core::{OmwuOomd, OptimizerEnum};
    use ndarray::array;

    fn make_opt() -> Optimizer {
        Optimizer {
            inner: OptimizerEnum::OmwuOomd(OmwuOomd::new(0.1, 2)),
        }
    }

    #[test]
    fn test_concentric_output_building() {
        let matrix = array![[0.5, 0.5], [0.5, 0.5]];
        let hyperparams = HyperParams {
            num_explo: 4,
            num_iter_per_explo: 5,
            inner_radius: 0.0,
            outer_radius: 0.2,
            metric_method: MetricMethodType::TotalVar,
        };
        
        let output = ConcentricOutput::new(matrix.view(), make_opt(), hyperparams, 2);
        
        assert_eq!(output.slices_boundaries.len(), 2);
        assert_eq!(output.metrics.len(), 2);
        assert_eq!(output.metrics[0].len(), 2); // 4 explo / 2 slices
    }
}
