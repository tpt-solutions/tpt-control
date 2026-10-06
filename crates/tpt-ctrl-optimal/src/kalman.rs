//! Steady-state Kalman filtering and the LQG separation principle.
//!
//! The steady-state Kalman gain is the dual of the DARE: the prediction
//! error covariance `P` solves the DARE on the transposed system
//! `(A^T, C^T, Q_noise, R_noise)`.

use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::Result;

use crate::dare::dare;
use crate::lqr::{lqr, LqrSolution};

/// Steady-state Kalman filter gains for `x+ = A x + w`, `y = C x + v` with
/// process covariance `Q_noise` and measurement covariance `R_noise`.
///
/// Returns `(P, L)` with `P` the prediction error covariance and `L` the
/// innovation gain `P C^T (C P C^T + R)^{-1}` (measurement update:
/// `x_hat+ = x_pred + L (y - C x_pred)`).
pub fn kalman_gain(
    a: &DMat<f64>,
    c: &DMat<f64>,
    q_noise: &DMat<f64>,
    r_noise: &DMat<f64>,
) -> Result<(DMat<f64>, DMat<f64>)> {
    let sol = dare(&a.transpose(), &c.transpose(), q_noise, r_noise)?;
    // Innovation gain L = P C^T (C P C^T + R)^{-1}.
    let s_cov = &(c * &sol.x) * &c.transpose() + r_noise;
    let l = &(&sol.x * &c.transpose()) * &s_cov.inverse()?;
    Ok((sol.x, l))
}

/// LQG controller: LQR state feedback plus a steady-state Kalman observer.
///
/// By the separation principle the two designs are computed independently
/// from the LQ cost `(Q_cost, R_cost)` and the noise covariances
/// `(Q_noise, R_noise)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LqgController {
    /// LQR gain.
    pub k: DMat<f64>,
    /// Kalman innovation gain.
    pub l: DMat<f64>,
    /// Riccati solution from the LQR design.
    pub x: DMat<f64>,
    /// Estimation error covariance from the Kalman design.
    pub p: DMat<f64>,
}

/// Designs the LQG controller for `x' = A x + B u`, `y = C x`.
pub fn lqg(
    a: &DMat<f64>,
    b: &DMat<f64>,
    c: &DMat<f64>,
    q_cost: &DMat<f64>,
    r_cost: &DMat<f64>,
    q_noise: &DMat<f64>,
    r_noise: &DMat<f64>,
) -> Result<LqgController> {
    let LqrSolution { x, k } = lqr(a, b, q_cost, r_cost)?;
    let (p, l) = kalman_gain(a, c, q_noise, r_noise)?;
    Ok(LqgController { k, l, x, p })
}

/// Discrete-time LQG for `x+ = A x + B u`, `y = C x` (DLQR plus the
/// discrete steady-state Kalman observer).
pub fn dlqg(
    a: &DMat<f64>,
    b: &DMat<f64>,
    c: &DMat<f64>,
    q_cost: &DMat<f64>,
    r_cost: &DMat<f64>,
    q_noise: &DMat<f64>,
    r_noise: &DMat<f64>,
) -> Result<LqgController> {
    let sol = crate::dlqr(a, b, q_cost, r_cost)?;
    let (p, l) = kalman_gain(a, c, q_noise, r_noise)?;
    Ok(LqgController {
        k: sol.k,
        l,
        x: sol.x,
        p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ctrl_core::analysis::is_stable_discrete;
    use tpt_ctrl_verify::testing::assert_mat_close;

    #[test]
    fn kalman_gain_matches_direct_recursion() {
        // Compare the DARE-based steady state against iterating the
        // prediction-covariance recursion to convergence.
        let a = DMat::from_rows(&[&[0.9, 0.4], &[-0.2, 0.7]]);
        let c = DMat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let qn = DMat::from_rows(&[&[0.5, 0.1], &[0.1, 0.3]]);
        let rn = DMat::identity(2) * 0.2;
        let (p, _) = kalman_gain(&a, &c, &qn, &rn).unwrap();

        let mut p_it = qn.clone();
        for _ in 0..2000 {
            // P = A P A^T - A P C^T (R + C P C^T)^-1 C P A^T + Q
            let cpc = &(&c * &p_it) * &c.transpose();
            let s = &(&cpc + &rn).inverse().unwrap();
            let cpa = &(&(&c * &p_it) * &a.transpose());
            let ap = &(&a * &p_it) * &a.transpose();
            let corr = &(&(&a * &p_it) * &c.transpose()) * s * cpa;
            let p_next = &(ap - &corr) + &qn;
            let change = (&p_next - &p_it).norm_fro() / p_next.norm_fro();
            p_it = p_next;
            if change < 1e-14 {
                break;
            }
        }
        assert_mat_close(&p, &p_it, 1e-8);
    }

    #[test]
    fn dlqg_separation_gives_stable_loop() {
        let a = DMat::from_rows(&[&[1.0, 0.1], &[0.0, 1.0]]);
        let b = DMat::from_rows(&[&[0.005], &[0.1]]);
        let c = DMat::from_rows(&[&[1.0, 0.0]]);
        let q_cost = DMat::identity(2);
        let r_cost = DMat::from_rows(&[&[1.0]]);
        let q_noise = DMat::identity(2);
        let r_noise = DMat::from_rows(&[&[0.1]]);
        let lqg = dlqg(&a, &b, &c, &q_cost, &r_cost, &q_noise, &r_noise).unwrap();
        // Both design loops stable.
        let cl = &a - &(&b * &lqg.k);
        assert!(is_stable_discrete(&cl, 1e-9).unwrap());
        let obs = &(&DMat::identity(2) - &(&lqg.l * c)) * &a;
        assert!(is_stable_discrete(&obs, 1e-9).unwrap());
    }
}
