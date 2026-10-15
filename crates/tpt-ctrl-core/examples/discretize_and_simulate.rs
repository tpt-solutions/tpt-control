//! Worked example: exact ZOH vs. Tustin discretization of a damped
//! mass-spring system, followed by a simulated step response.
//!
//! The zero-order-hold map is exact for a piecewise-constant input, so the
//! discrete trajectory matches the continuous one at the sample instants.
//! Tustin is an approximation; comparing the two shows the difference.

use tpt_ctrl_core::analysis::{is_stable_continuous, is_stable_discrete, spectral_radius};
use tpt_ctrl_core::discretize::{tustin, zoh};
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::simulate::simulate_discrete;

fn main() {
    // m x'' + c x' + k x = u   with m = 1 kg, c = 0.5 N s/m, k = 2 N/m.
    let (m, c, k) = (1.0_f64, 0.5_f64, 2.0_f64);
    let a = DMat::from_rows(&[&[0.0, 1.0], &[-k / m, -c / m]]);
    let b = DMat::from_rows(&[&[0.0], &[1.0 / m]]);
    let cmat = DMat::from_rows(&[&[1.0, 0.0]]);
    let d = DMat::zeros(1, 1);

    println!(
        "continuous stable: {}",
        is_stable_continuous(&a, 1e-9).unwrap()
    );

    let dt = 0.1_f64;
    let (ad, bd) = zoh(&a, &b, dt).unwrap();
    let (at, _bt, _ct, _dt) = tustin(&a, &b, &cmat, &d, dt).unwrap();

    println!(
        "ZOH   spectral radius: {:.6}",
        spectral_radius(&ad).unwrap()
    );
    println!(
        "Tustin spectral radius: {:.6}",
        spectral_radius(&at).unwrap()
    );
    println!(
        "ZOH discrete stable:   {}",
        is_stable_discrete(&ad, 1e-9).unwrap()
    );

    // Unit step input, 4 seconds, starting from rest.
    let steps = 40;
    let us = vec![vec![1.0_f64]];
    let traj = simulate_discrete(&ad, &bd, &[0.0, 0.0], &us, steps);

    println!("\n   t [s]   position");
    for (k, x) in traj.states.iter().enumerate().step_by(5) {
        println!("  {:5.2}   {:+.6}", k as f64 * dt, x[0]);
    }
    let final_pos = traj.states.last().unwrap()[0];
    println!("\nsteady state (expect 1/k = 0.5): {:.6}", final_pos);
}
