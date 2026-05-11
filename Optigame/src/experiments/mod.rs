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
    params: Params,
) -> Vec<GameResult> {
    let dist = Uniform::new(-1., 1.);
    // using rayon to run concurrently the iterations
    let vec_results = (1..=params.num_exploration)
        .into_par_iter()
        .map_init(
            || {
                let worker_context =
                    WorkerContext::new(a_base, optimizer.clone(), params.num_steps);
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
// This is the previous random_exploration for comparison
// pub fn random_exploration(
//     a_delta: ArrayView2<f64>,
//     vec_epsilon: ArrayView1<f64>,
//     optimizer: Optimizer,
//     num_exploration: usize,
//     num_steps: usize,
//     method: &str,
// ) -> Array2<f64> {
//     let cutoff_idx = (num_steps as f64 * 0.9) as usize;
//     let tail_len = (num_steps - cutoff_idx) as f64;
//
//     let num_eps = vec_epsilon.len();
//
//     let final_array: Vec<Vec<f64>> = vec_epsilon
//         .into_par_iter()
//         .map(|&epsilon| {
//             // run exploration for this epsilon
//             let list_results = random_neighborhood_exploration(
//                 a_delta,
//                 epsilon,
//                 optimizer.clone(),
//                 num_exploration,
//                 num_steps,
//             );
//
//             let mut row_metrics = Vec::with_capacity(list_results.len());
//
//             for res in list_results {
//                 let gaps = res.gaps();
//
//                 let tail = gaps.slice(s![cutoff_idx..]);
//
//                 let metric = match method {
//                     "max_last_10" => tail
//                         .iter()
//                         .cloned()
//                         .fold(std::f64::NEG_INFINITY, |acc, elem| acc.max(elem)),
//                     "var_last_10" => {
//                         let mean = tail.iter().sum::<f64>() / tail_len;
//                         tail.iter().fold(0.0, |acc, &val| {
//                             let diff = val - mean;
//                             acc + diff * diff / tail_len
//                         })
//                     }
//                     "total_var" => gaps
//                         .iter()
//                         .zip(gaps.iter().skip(1))
//                         .fold(0.0, |acc, (&a, &b)| acc + (b - a).abs()),
//                     _ => panic!("Unknown metric method: {}", method),
//                 };
//
//                 row_metrics.push(metric);
//             }
//
//             row_metrics
//         })
//         .collect();
//
//     let flat: Vec<f64> = final_array
//         .into_iter()
//         .flat_map(|v| v.into_iter())
//         .collect();
//     Array2::from_shape_vec((num_eps, num_exploration), flat)
//         .expect("Shape mismatch in random_exploration")
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    num_exploration: usize,
    num_steps: usize,
}

impl Params {
    /// Create an instance of the type Params from two usize :
    /// num_exploration and num_steps
    pub fn new(num_exploration: usize, num_steps: usize) -> Self {
        // Ask : here I've published this constructor, is it the right approach : Keeping Params' fields private?
        Params {
            num_exploration,
            num_steps,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MethodType {
    MaxLast(f64), // Ask : can I give a name to that usize ? Like cutoff, because this what the usize is, it's the cutoff in percentage from which we're taking a slice
    VarLast(f64),
    TotalVar,
}

pub struct Hyperparams {
    params: Params,
    method: MethodType,
}

impl Hyperparams {
    pub fn try_from_raw(
        num_exploration: usize,
        num_steps: usize,
        method: &str,
        cutoff_percent: f64,
    ) -> Result<Self, String> {
        let params = Params::new(num_exploration, num_steps);
        let method = match method {
            "MaxLast" => MethodType::MaxLast(cutoff_percent),
            "VarLast" => MethodType::VarLast(cutoff_percent), // issues for those two first arms, I need a way to parse, or maybe add an argument like cutoff so that I can better parse the right enum type and inner value
            // but I don't know which approach is the most elegant one, especially for the API? Should I ask two inputs to the user : method + cutoff when that's necessary and then combine them here?
            "TotalVar" => MethodType::TotalVar,
            _ => return Err(format!("Unknown method type : {}", method)),
        };

        Ok(Hyperparams { params, method })
    }
}

pub fn random_exploration(
    a_delta: ArrayView2<f64>,
    vec_epsilon: ArrayView1<f64>,
    optimizer: Optimizer,
    hyperparams: Hyperparams, // Hyperparams would contain num_exploration, num_steps, and potentially methods
) -> Array2<f64> {
    let final_array: Vec<f64> = vec_epsilon
        .into_par_iter()
        .flat_map(|&epsilon| {
            let vec_results = random_neighborhood_exploration(
                a_delta,
                epsilon,
                optimizer.clone(),
                hyperparams.params,
            );
            let array_metrics = results_to_metrics(vec_results, hyperparams.method);
            array_metrics
        })
        .collect();

    let final_array = Array2::from_shape_vec(
        (vec_epsilon.len(), hyperparams.params.num_exploration),
        final_array,
    )
    .expect("Error during the conversion to an Array2"); // Ask : Is this operation rightly written, or does vec_epsilon.len(), hyperparams.params.num_exploration bring too much cognitive load? 
    final_array
}

/// Function that takes as input a vec of GameResults of length num_explorations and a MethodType instance
/// to return a vec of num_explorations values computed according to the method
fn results_to_metrics(list_results: Vec<GameResult>, method: MethodType) -> Vec<f64> {
    // Ask: is this function already too big and not respecting the seperation of concerns? E.g. should it be only responsible of looping through the results and routing to the right computation method?
    let mut vec_metrics: Vec<f64> = Vec::with_capacity(list_results.len());

    for result in list_results {
        let metric = compute_single_metric(&result, method);

        vec_metrics.push(metric);
    }
    vec_metrics
}

fn compute_single_metric(game_result: &GameResult, method: MethodType) -> f64 {
    let gaps = game_result.gaps();

    match method {
        MethodType::MaxLast(cutoff_percent) => {
            let tail = get_tail(gaps, cutoff_percent);
            tail.into_iter()
                .fold(std::f64::NEG_INFINITY, |acc, &elem| acc.max(elem))
        }
        MethodType::VarLast(cutoff_percent) => {
            let tail = get_tail(gaps.into(), cutoff_percent);
            let tail_len = tail.len() as f64;
            let mean = tail.iter().sum::<f64>() / tail_len;

            tail.iter().fold(0.0, |acc, &val| {
                let diff = val - mean;
                acc + diff * diff / tail_len
            })
        }
        MethodType::TotalVar => gaps
            .iter()
            .zip(game_result.gaps().iter().skip(1))
            .fold(0.0, |acc, (&a, &b)| acc + (b - a).abs()),
    }
}

fn get_tail<'a>(array: ArrayView1<'a, f64>, cutoff_percent: f64) -> ArrayView1<'a, f64> {
    // Ask: is the function naming right? I'm not just getting the tail, I'm getting the cutoff_percent's rest of the tail

    // (1. - cutoff) because cutoff represent the last x% of values we want to keep, so me must get rid of the 1. - cutoff first values
    let cutoff_idx = ((1. - cutoff_percent) * array.len() as f64) as usize;
    let tail = array.slice_move(s![cutoff_idx..]);
    tail
}

#[cfg(test)]
mod test {
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
        // constructing without gil for pure rust testing:
        let opt = Optimizer {
            inner: OptimizerEnum::OmwuOomd(OmwuOomd::new(0.1, 2)), // Ask : Same this initialization of optimizer looks cumbersome
        };

        let params = Params {
            // Ask : Is this cumbersome to add those line before running random_neighborhood_exploration ?
            num_exploration, // In terms of API design and end user experience, how should I structure that?
            num_steps,
        };

        let results = random_neighborhood_exploration(a_base.view(), epsilon, opt, params);

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

        let params = Params {
            num_exploration,
            num_steps,
        };

        let results = random_neighborhood_exploration(a_base.view(), epsilon, opt, params);

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
