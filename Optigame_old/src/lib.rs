mod math;
mod optimizers;
mod experiments;

use ndarray::array;
use rayon::vec;
use crate::math::{M, S};
use crate::optimizers::{Ogda, OmwuOftrl, OmwuOomd, Optimizer};
use crate::experiments::{GameState, Experiment, Neighborhood, run_neighborhood_exploration, save_sweep_to_npz};

fn main() {
    let delta = 0.01;
    let a_base: M = array![
        [1., delta],
        [1.-delta, 1.]
    ];
    let x_init: S = S::from_projected(array![0.5, 0.5]);
    let y_init: S = S::from_projected(array![0.5, 0.5]);
    let num_steps = 400_000;
    let eta = 0.01;

    let state = GameState::new(x_init, y_init, a_base.clone());

    let optimizers: Vec<Optimizer> = vec![Optimizer::OmwuOomd(OmwuOomd::new(eta, 2)),
                                          Optimizer::OmwuOftrl(OmwuOftrl::new(eta, 2)),
                                          Optimizer::Ogda(Ogda::new(eta, 2))];


    println!("Generating neighborhood with PCA...");
    let num_perturbations = 1000;
    let neighborhood = Neighborhood::new(&a_base, num_perturbations);

    println!("Running parallel exploration of {} perturbations...", num_perturbations);
    let (pca_coords, distances) = run_neighborhood_exploration(neighborhood, optimizers, num_steps);

    // Construct a structured directory tree string
    let output_file = format!("../data/sweeps/delta_{}/omwu_eta_{}/steps_{}/sweep_results.npz", delta, eta, num_steps);
    
    println!("Saving sweep results to {}...", output_file);
    save_sweep_to_npz(&output_file, &pca_coords, &distances).expect("Failed to save sweep results");
    
    println!("Done!");
}
