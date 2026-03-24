mod math;
mod optimizers;
mod experiments;

use std::vec;

use ndarray::array;

use crate::math::{M, S};

use crate::optimizers::{Ogda, OmwuOomd, OmwuOftrl, Optimizer};

use crate::experiments::{GameState, Experiment};

fn main() {
    let a_delta: M = array![
        [0.5 + 0.01, 0.5],
        [0., 1.]
    ];
    let x_init: S = S::from_projected(array![0.5, 0.5]);
    let y_init: S = S::from_projected(array![0.5, 0.5]);
    let num_steps = 400_000;

    let rps_state = GameState::new(x_init, y_init, a_delta);

    let optimizer = OmwuOomd::new(
        0.01,
        S::from_projected(array![0., 0.]),
        S::from_projected(array![0., 0.]),
    );

    let experiment = Experiment::new(rps_state, Optimizer::OmwuOomd(optimizer), num_steps);

    let rps_state_final = experiment.run_experiment();

    rps_state_final.save_to_npz("../Scripts/test.npz").expect("Problem writing to file");

    println!("Final Strategy for x: {:#?}", rps_state_final.x_history().last());
    println!("Final Strategy for y: {:#?}", rps_state_final.y_history().last());
    println!("Final Duality gap: {:#?}", rps_state_final.gaps().last());
}
