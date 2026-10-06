//! Worked example: LQR stabilization of a linearized inverted pendulum.
//!
//! Designs a state-feedback gain for the classic cart-pole, verifies the
//! closed loop is Hurwitz, and simulates the recovery from a tilt.

use tpt_ctrl_core::analysis::{is_controllable, is_stable_continuous};
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::lqr;

fn main() {
    // Cart-pole: M = 1 kg cart, m = 0.1 kg pole, l = 0.5 m, g = 9.8.
    let (m, m_cart, l, g) = (0.1_f64, 1.0_f64, 0.5_f64, 9.8_f64);
    let a = DMat::from_rows(&[
        &[0.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, -(m * g) / m_cart, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0, ((m_cart + m) * g) / (m_cart * l), 0.0],
    ]);
    let b = DMat::from_rows(&[&[0.0], &[1.0 / m_cart], &[0.0], &[-1.0 / (m_cart * l)]]);

    println!("controllable: {}", is_controllable(&a, &b, 1e-9).unwrap());

    // Weights: care about the pole angle most.
    let q = DMat::diagonal(&[1.0, 1.0, 10.0, 1.0]);
    let r = DMat::from_rows(&[&[1.0]]);
    let sol = lqr(&a, &b, &q, &r).unwrap();
    println!("LQR gain K = {:?}", sol.k);

    let cl = &a - &(&b * &sol.k);
    println!(
        "closed loop stable: {}",
        is_stable_continuous(&cl, 1e-9).unwrap()
    );

    // Simulate recovery from a 0.2 rad tilt with RK4 on the linear model.
    let cl_dyn = |_t: f64, x: &[f64], _u: &[f64], out: &mut [f64]| {
        for i in 0..4 {
            let mut acc = 0.0;
            for j in 0..4 {
                acc += cl[(i, j)] * x[j];
            }
            out[i] = acc;
        }
    };
    let mut x = vec![0.0, 0.0, 0.2, 0.0];
    let empty: [f64; 0] = [];
    let dt = 0.01;
    for k in 0..600 {
        tpt_ctrl_core::simulate::rk4_step(&cl_dyn, k as f64 * dt, dt, &mut x, &empty);
        if k % 100 == 0 {
            println!("t = {:5.2} s   angle = {:+.6} rad", k as f64 * dt, x[2]);
        }
    }
    println!("final angle: {:+.9} rad", x[2]);
}
