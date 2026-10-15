//! Quickstart: one tour of the umbrella crate.
//!
//! Designs an LQR state feedback for a double integrator, verifies the
//! closed loop, then runs a Kalman filter on noisy position measurements.
//! Run with `cargo run -p tpt-control --example quickstart`.

use tpt_control::prelude::*;

fn main() {
    // 1. Plant: x'' = u, state = [position, velocity].
    let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
    let b = DMat::from_rows(&[&[0.0], &[1.0]]);

    // 2. Design: LQR minimizing  int x'Qx + u'Ru dt.
    let q = DMat::identity(2);
    let r = DMat::from_rows(&[&[1.0]]);
    let sol = lqr(&a, &b, &q, &r).expect("LQR design");
    println!("LQR gain K = {:?}", sol.k);

    // 3. Verify: the closed loop A - B K must be Hurwitz.
    let acl = &a - &(&b * &sol.k);
    let stable = tpt_control::tpt_ctrl_core::analysis::is_stable_continuous(&acl, 1e-9).unwrap();
    println!("closed loop stable: {stable}");

    // 4. Estimate: discrete Kalman filter on a noisy position sensor.
    let dt = 0.1;
    let ad = DMat::from_rows(&[&[1.0, dt], &[0.0, 1.0]]);
    let h = DMat::from_rows(&[&[1.0, 0.0]]);
    let mut kf = kf::KalmanFilter::new(
        DMat::from_rows(&[&[0.0], &[0.0]]),
        DMat::identity(2),
        ad,
        DMat::identity(2) * 1e-4,
        h,
        DMat::from_rows(&[&[0.05]]),
    );
    for k in 1..=50 {
        kf.predict();
        let truth = 0.5 * k as f64 * dt;
        let z = truth + if k % 2 == 0 { 0.05 } else { -0.05 };
        kf.update(&DMat::from_rows(&[&[z]])).unwrap();
    }
    println!("filtered position: {:.4}", kf.x[(0, 0)]);
    println!("filtered velocity: {:.4} (truth 0.5)", kf.x[(1, 0)]);
}
