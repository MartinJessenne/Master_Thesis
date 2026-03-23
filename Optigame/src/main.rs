mod math;
mod optimizers;
mod experiments;

use std::vec;

use ndarray::array;

use crate::math::{M, S};

use crate::optimizers::{OGDA, OMWU};

use crate::experiments::{InitialState, Experiment};

fn main() {
    let A_delta: M = array![
        [0.5 + 0.01, 0.5],
        [0., 1.]
    ];
    let x_init: S = S::from_projected(array![0.5, 0.5]);
    let y_init: S = S::from_projected(array![0.5, 0.5]);
    let eta = 0.01;
    let num_steps = 4_000_000;

    let rps_state = InitialState::new(x_init.clone(), y_init.clone(), x_init, y_init, A_delta, eta);

    let optimizer = OMWU{
        grad_x_prev: array![0., 0.],
        grad_y_prev: array![0., 0.],

    };

    let experiment = Experiment {state: rps_state, optimizer, num_steps};

    let rps_state_final = run_experiment(rps_state, optimizer, num_steps);

    rps_state_final.save_to_npz("test.npz").expect("Problem writing to file");

    println!("Final Strategy for x: {:#?}", rps_state_final.x());
    println!("Final Strategy for y: {:#?}", rps_state_final.y());
    println!("Final Duality gap: {:#?}", rps_state_final.gaps().last());
}
