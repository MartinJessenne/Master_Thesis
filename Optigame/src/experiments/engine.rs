use crate::domain::structure::{Experiment, GameState};
use super::types::RandomGameResult;
use crate::optimizers::core::{Optimizer, OptimizerStrategy};
use ndarray::ArrayView2;
use rand::{
    distr::{Bernoulli, Distribution, Uniform},
    rngs::ThreadRng,
};
use rayon::prelude::*;

pub(crate) struct Exploration<'a> {
    matrix: ArrayView2<'a, f64>,
    state: GameState,
    optimizer: Optimizer,
    num_steps: usize,
    distribution: Uniform<f64>,
    rng: ThreadRng,
    norm_fn: fn(ArrayView2<f64>) -> f64,
}

impl<'a> Exploration<'a> {
    pub(crate) fn context(
        matrix: ArrayView2<'a, f64>,
        optimizer: Optimizer,
        num_steps: usize,
        unif: Uniform<f64>,
        norm_fn: fn(ArrayView2<f64>) -> f64,
    ) -> Self {
        let state = GameState::from_matrix(matrix.into_owned());
        let rng = rand::rng();
        Exploration {
            matrix,
            state,
            optimizer,
            num_steps,
            distribution: unif,
            rng,
            norm_fn,
        }
    }

    pub(crate) fn execute_clean_run(&mut self) -> RandomGameResult {
        self.reset_state();
        self.apply_perturbation();
        self.run_experiment()
    }

    fn reset_state(&mut self) {
        self.optimizer.reset();
        self.state.reset_strats();
    }

    fn apply_perturbation(&mut self) {
        let unif = self.distribution;
        let bern = Bernoulli::new(0.5).unwrap();
        self.state.a.mapv_inplace(|_| {
            let val = unif.sample(&mut self.rng);
            if bern.sample(&mut self.rng) { -val } else { val }
        });
    }

    fn run_experiment(&mut self) -> RandomGameResult {
        let norm = (self.norm_fn)(self.state.a.view());
        self.state.a += &self.matrix;
        let experiment_runner = Experiment::new(self.num_steps);
        let game_result = experiment_runner
            .run_experiment_until_convergence(&mut self.state, &mut self.optimizer);
        RandomGameResult { norm, game_result }
    }
}

pub(crate) fn run_batch_exploration(
    matrix: ArrayView2<f64>,
    optimizer: Optimizer,
    unif: Uniform<f64>,
    num_exploration: usize,
    num_iter_per_explo: usize,
    norm_fn: fn(ArrayView2<f64>) -> f64,
) -> Vec<RandomGameResult> {
    (0..num_exploration)
        .into_par_iter()
        .map_init(
            || Exploration::context(matrix, optimizer.clone(), num_iter_per_explo, unif, norm_fn),
            |exploration_context, _| exploration_context.execute_clean_run(),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::maths::max_norm;
    use crate::optimizers::core::{OmwuOomd, OptimizerEnum};
    use approx::assert_relative_eq;
    use ndarray::array;

    fn make_opt() -> Optimizer {
        Optimizer {
            inner: OptimizerEnum::OmwuOomd(OmwuOomd::new(0.1, 2)),
        }
    }

    #[test]
    fn test_exploration_structural_integrity() {
        let num_exploration = 5;
        let num_steps = 10;
        let a_base = array![[0.5, 0.5], [0.5, 0.5]];
        let opt = make_opt();
        let unif = Uniform::new(-0.1, 0.1).unwrap();
        let results = run_batch_exploration(a_base.view(), opt, unif, num_exploration, num_steps, max_norm);
        assert_eq!(results.len(), num_exploration);
        assert_eq!(results[0].game_result.gaps_history.len(), num_steps);
    }

    #[test]
    fn test_exploration_zero_noise_determinism() {
        let num_exploration = 10;
        let num_steps = 20;
        let a_base = array![[0.8, 0.2], [0.3, 0.7]];
        let opt = make_opt();
        let unif = Uniform::new_inclusive(0.0, 0.0).unwrap();
        let results = run_batch_exploration(a_base.view(), opt, unif, num_exploration, num_steps, max_norm);
        let baseline_gaps = &results[0].game_result.gaps_history;
        for res in results.iter().skip(1) {
            for (val1, val2) in baseline_gaps.iter().zip(res.game_result.gaps_history.iter()) {
                assert_relative_eq!(val1, val2, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_worker_perturbation_bounds() {
        let num_steps = 1;
        let epsilon = 0.1;
        let a_base = array![[0.5, 0.5], [0.5, 0.5]];
        let opt = make_opt();
        let unif = Uniform::new(0.0, epsilon).unwrap();
        let mut worker = Exploration::context(a_base.view(), opt, num_steps, unif, max_norm);
        let _result = worker.execute_clean_run();
        for (original, perturbed) in a_base.iter().zip(worker.state.a.iter()) {
            let diff = (perturbed - original).abs();
            assert!(diff <= epsilon + 1e-12);
        }
    }
}
