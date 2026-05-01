use crate::domain::{Experiment, GameResult, GameState};
use crate::optimizers::{Optimizer, OptimizerStrategy};
use ndarray::{Array2, ArrayView1, ArrayView2, s};
use ndarray_rand::{
    rand::{rngs::ThreadRng, thread_rng},
    rand_distr::{Distribution, Uniform},
};
use rayon::prelude::*;

pub struct WorkerContext {
    pub state: GameState,
    pub opt: Optimizer,
    pub exp: Experiment,
    pub rng: ThreadRng,
}

impl WorkerContext {
    pub fn new(a_base: ArrayView2<f64>, optimizer: Optimizer, num_steps: usize) -> Self {
        let dummy = Array2::zeros(a_base.dim());
        let state = GameState::from_matrix(dummy);
        let opt = optimizer.clone();
        let exp = Experiment::new(num_steps);
        let rng = thread_rng();
        WorkerContext {
            state,
            opt,
            exp,
            rng,
        }
    }

    pub fn run(&mut self, epsilon: f64, a_base: ArrayView2<f64>, dist: Uniform<f64>) -> GameResult {
        // Reset the optimizer
        self.opt.reset();

        // fill in the matrix and initial states, 0 allocation
        self.state.reset_strats();
        self.state.a.mapv_inplace(|_| dist.sample(&mut self.rng));

        // perform the perturbation
        self.state.a *= epsilon;
        self.state.a += &a_base;

        // run the experiment
        let game_result = self
            .exp
            .run_experiment_until_convergence(&mut self.state, &mut self.opt);

        game_result
    }
}

pub fn random_neighborhood_exploration(
    a_base: ArrayView2<f64>,
    epsilon: f64,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
) -> Vec<GameResult> {
    let dist = Uniform::new(-1., 1.);
    // using rayon to run concurrently the iterations
    let vec_results = (1..=num_exploration)
        .into_par_iter()
        .map_init(
            || {
                let worker_context = WorkerContext::new(a_base, optimizer.clone(), num_steps);
                return worker_context;
            },
            |worker_context, _index| {
                let game_result = worker_context.run(epsilon, a_base, dist);
                return game_result;
            },
        )
        .collect();

    vec_results
}

/// Conducts a grid search across an array of epsilon values, for each epsilon, it performs num steps iterations of
/// random_neighborhood_exploration and contracts the list of results to a single scalar according to three different methods :
/// 1. max_last_10
/// 2. var_last_10
/// 3. total_var
pub fn random_exploration(
    a_delta: ArrayView2<f64>,
    vec_epsilon: ArrayView1<f64>,
    optimizer: Optimizer,
    num_exploration: usize,
    num_steps: usize,
    method: &str,
) -> Array2<f64> {
    let cutoff_idx = (num_steps as f64 * 0.9) as usize;
    let tail_len = (num_steps - cutoff_idx) as f64;

    let num_eps = vec_epsilon.len();

    let final_array: Vec<Vec<f64>> = vec_epsilon
        .into_par_iter()
        .map(|&epsilon| {
            // run exploration for this epsilon
            let list_results = random_neighborhood_exploration(
                a_delta,
                epsilon,
                optimizer.clone(),
                num_exploration,
                num_steps,
            );

            let mut row_metrics = Vec::with_capacity(list_results.len());

            for res in list_results {
                let gaps = res.gaps();

                let tail = gaps.slice(s![cutoff_idx..]);

                let metric = match method {
                    "max_last_10" => tail
                        .iter()
                        .cloned()
                        .fold(std::f64::NEG_INFINITY, |acc, elem| acc.max(elem)),
                    "var_last_10" => {
                        let mean = tail.iter().sum::<f64>() / tail_len;
                        tail.iter().fold(0.0, |acc, &val| {
                            let diff = val - mean;
                            acc + diff * diff / tail_len
                        })
                    }
                    "total_var" => gaps
                        .iter()
                        .zip(gaps.iter().skip(1))
                        .fold(0.0, |acc, (&a, &b)| acc + (b - a).abs()),
                    _ => panic!("Unknown metric method: {}", method),
                };

                row_metrics.push(metric);
            }

            row_metrics
        })
        .collect();

    let flat: Vec<f64> = final_array
        .into_iter()
        .flat_map(|v| v.into_iter())
        .collect();
    Array2::from_shape_vec((num_eps, num_exploration), flat)
        .expect("Shape mismatch in random_exploration")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{OmwuOomd, OptimizerEnum};
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_exploration_structural_integrity() {
        let num_exploration = 5;
        let num_steps = 10;
        let epsilon = 0.1;
        let a_base = array![[0.5, 0.5], [0.5, 0.5]];
        // Constructing without GIL for pure rust testing:
        let opt = Optimizer {
            inner: OptimizerEnum::OmwuOomd(OmwuOomd::new(0.1, 2)),
        };

        let results = random_neighborhood_exploration(
            a_base.view(),
            epsilon,
            opt,
            num_exploration,
            num_steps,
        );

        assert_eq!(results.len(), num_exploration);
        assert_eq!(results[0].gaps_history.len(), num_steps);
        assert_eq!(results[0].x_history.nrows(), num_steps);
        assert_eq!(results[0].y_history.nrows(), num_steps);
    }

    #[test]
    fn test_exploration_zero_noise_determinism() {
        let num_exploration = 10;
        let num_steps = 20;
        let epsilon = 0.0; // Zero noise means all runs should be identical
        let a_base = array![[0.8, 0.2], [0.3, 0.7]];
        let opt = Optimizer {
            inner: OptimizerEnum::OmwuOomd(OmwuOomd::new(0.1, 2)),
        };

        let results = random_neighborhood_exploration(
            a_base.view(),
            epsilon,
            opt,
            num_exploration,
            num_steps,
        );

        assert_eq!(results.len(), num_exploration);

        let baseline_gaps = &results[0].gaps_history;

        // If epsilon is 0, every thread should compute the exact same trajectory.
        // This proves that worker state recycling is perfectly clean.
        for res in results.iter().skip(1) {
            for (val1, val2) in baseline_gaps.iter().zip(res.gaps_history.iter()) {
                assert_relative_eq!(val1, val2, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_worker_perturbation_bounds() {
        let num_steps = 1;
        let epsilon = 0.1;
        let a_base = array![[0.5, 0.5], [0.5, 0.5]];
        let opt = Optimizer {
            inner: OptimizerEnum::OmwuOomd(OmwuOomd::new(0.1, 2)),
        };

        // 1. Instantiate the isolated WorkerContext
        let mut worker = WorkerContext::new(a_base.view(), opt, num_steps);
        let dist = Uniform::new(-1.0, 1.0);

        // 2. Execute a single run
        let _result = worker.run(epsilon, a_base.view(), dist);

        // 3. Introspect the internal state to verify the perturbation math
        let perturbed_matrix = &worker.state.a;

        for (original, perturbed) in a_base.iter().zip(perturbed_matrix.iter()) {
            let diff = (perturbed - original).abs();
            // Since U is in [-1.0, 1.0], epsilon * U is in [-epsilon, epsilon]
            // We add a tiny float tolerance (1e-12) to prevent precision failures
            assert!(
                diff <= epsilon + 1e-12,
                "Perturbation difference {} exceeded epsilon {}",
                diff,
                epsilon
            );
        }
    }
}
