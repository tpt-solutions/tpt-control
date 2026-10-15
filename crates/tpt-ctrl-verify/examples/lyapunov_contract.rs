//! Worked example: checking a Lyapunov contract numerically.
//!
//! The LQR Riccati solution `X` of a stabilizable plant is a Lyapunov
//! function for the closed loop. This example samples `V(x) = x' X x` and
//! its derivative over a box of states to confirm `V > 0` and `V' < 0`.
//! A sampled check is evidence over a region, not a global proof.

use tpt_ctrl_core::analysis::is_stable_continuous;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_optimal::lqr;
use tpt_ctrl_verify::lyapunov::{
    check_lyapunov_contract, p_is_positive_definite, QuadraticWithField,
};

fn main() {
    // Lightly damped oscillator, stabilized with full-state LQR feedback.
    let a = DMat::from_rows(&[&[0.0, 1.0], &[-2.0, -0.4]]);
    let b = DMat::from_rows(&[&[0.0], &[1.0]]);
    let q = DMat::identity(2);
    let r = DMat::from_rows(&[&[1.0]]);

    let sol = lqr(&a, &b, &q, &r).unwrap();
    let p = &sol.x;
    let acl = &a - &(&b * &sol.k);
    println!(
        "closed loop Hurwitz: {}",
        is_stable_continuous(&acl, 1e-9).unwrap()
    );
    println!("P positive definite:  {}", p_is_positive_definite(p, 1e-12));

    // Closed-loop vector field x' = (A - B K) x, evaluated for the checker.
    let field = |x: &[f64], out: &mut [f64]| {
        for i in 0..2 {
            out[i] = (0..2).map(|j| acl[(i, j)] * x[j]).sum();
        }
    };
    let lyap = QuadraticWithField::new(p, field);

    // Sample a 4 x 4 box around the origin, skipping the origin itself.
    let bounds = [(-2.0, 2.0), (-2.0, 2.0)];
    match check_lyapunov_contract(&lyap, &bounds, 9, 0.0, 0.0) {
        Ok(report) => {
            println!("samples checked:      {}", report.samples);
            println!("min V(x) on box:      {:.6}", report.min_v);
            println!("max V'(x) on box:     {:.6}", report.max_v_dot);
        }
        Err(msg) => println!("contract violated: {msg}"),
    }
}
