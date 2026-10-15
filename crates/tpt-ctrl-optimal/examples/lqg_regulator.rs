//! Worked example: LQG regulator for a double integrator with a noisy
//! position sensor.
//!
//! LQR and the steady-state Kalman filter are designed independently (the
//! separation principle). The example checks both closed-loop matrices:
//! the regulator `A - B K` and the observer error dynamics `A - L C`.

use tpt_ctrl_core::analysis::{is_stable_continuous, poles};
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::lqg;

fn main() {
    // x = [position, velocity], u = force, y = position (noisy).
    let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
    let b = DMat::from_rows(&[&[0.0], &[1.0]]);
    let c = DMat::from_rows(&[&[1.0, 0.0]]);

    // Cost: penalize position error 10x more than velocity, and effort lightly.
    let q_cost = DMat::diagonal(&[10.0, 1.0]);
    let r_cost = DMat::from_rows(&[&[0.1]]);

    // Noise: process disturbance on velocity, measurement noise sigma = 0.1.
    let q_noise = DMat::diagonal(&[0.0, 0.5]);
    let r_noise = DMat::from_rows(&[&[0.01]]);

    let ctrl = lqg(&a, &b, &c, &q_cost, &r_cost, &q_noise, &r_noise).unwrap();

    println!(
        "LQR gain K = [{:.4}, {:.4}]",
        ctrl.k[(0, 0)],
        ctrl.k[(0, 1)]
    );
    println!(
        "Kalman gain L = [{:.4}, {:.4}]^T",
        ctrl.l[(0, 0)],
        ctrl.l[(1, 0)]
    );

    let regulator = &a - &(&b * &ctrl.k);
    let observer_error = &a - &(&ctrl.l * &c);
    println!(
        "regulator stable (A - BK):       {}",
        is_stable_continuous(&regulator, 1e-9).unwrap()
    );
    println!(
        "observer stable (A - LC):        {}",
        is_stable_continuous(&observer_error, 1e-9).unwrap()
    );

    println!("\nregulator poles:");
    for p in poles(&regulator).unwrap() {
        println!("  {:+.4} {:+.4}i", p.re, p.im);
    }
}
