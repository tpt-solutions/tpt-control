//! MATLAB Control System Toolbox look-alikes.
//!
//! Each function mirrors a MATLAB name and its output order, so scripts
//! translate line for line. See `docs/MATLAB.md` for the full mapping.
//!
//! | MATLAB                 | here                         | returns                    |
//! |------------------------|------------------------------|----------------------------|
//! | `[A,B,C,D] = tf2ss(n,d)` | `tf2ss`                  | `(A, B, C, D)`             |
//! | `[Ad,Bd] = c2d(A,B,Ts)`  | `c2d`                    | `(Ad, Bd)` (exact ZOH)     |
//! | `[K,S,e] = lqr(A,B,Q,R)` | `lqr`                    | `(K, S, e)`                |
//! | `[K,S,e] = dlqr(A,B,Q,R)`| `dlqr`                   | `(K, S, e)`                |
//! | `[L,P] = kalman(A,C,Qn,Rn)` | `kalman`              | `(L, P)`                   |
//!
//! `e` is the vector of closed-loop eigenvalues of `A − BK`, as in MATLAB.

use std::vec::Vec;

use tpt_ctrl_classic::tf::{TfError, TransferFunction};
use tpt_ctrl_core::analysis::poles;
use tpt_ctrl_core::c64::C64;
use tpt_ctrl_core::discretize::zoh;
use tpt_ctrl_core::dmat::DMat;
use tpt_ctrl_core::error::LinalgError;
use tpt_ctrl_optimal::{dlqr as dlqr_solve, kalman_gain, lqr as lqr_solve};

/// `(A, B, C, D)` state-space matrices.
pub type StateSpace4 = (DMat<f64>, DMat<f64>, DMat<f64>, DMat<f64>);
/// `(K, S, e)`: gain, Riccati solution, closed-loop eigenvalues.
pub type LqrTriple = (DMat<f64>, DMat<f64>, Vec<C64>);

/// `[A,B,C,D] = tf2ss(num, den)`: controllable-canonical realization of a
/// SISO transfer function (coefficients in descending powers of `s`).
pub fn tf2ss(num: &[f64], den: &[f64]) -> Result<StateSpace4, TfError> {
    TransferFunction::new(num, den).map(|tf| tf.to_state_space())
}

/// `[Ad,Bd] = c2d(A,B,Ts)`: exact zero-order-hold discretization.
pub fn c2d(a: &DMat<f64>, b: &DMat<f64>, ts: f64) -> Result<(DMat<f64>, DMat<f64>), LinalgError> {
    zoh(a, b, ts)
}

/// `[K,S,e] = lqr(A,B,Q,R)`: continuous-time LQR gain, Riccati solution `S`
/// and closed-loop eigenvalues `e`.
pub fn lqr(
    a: &DMat<f64>,
    b: &DMat<f64>,
    q: &DMat<f64>,
    r: &DMat<f64>,
) -> Result<LqrTriple, LinalgError> {
    let sol = lqr_solve(a, b, q, r)?;
    let e = closed_loop_eigs(a, b, &sol.k)?;
    Ok((sol.k, sol.x, e))
}

/// `[K,S,e] = dlqr(A,B,Q,R)`: discrete-time LQR gain, Riccati solution `S`
/// and closed-loop eigenvalues `e`.
pub fn dlqr(
    a: &DMat<f64>,
    b: &DMat<f64>,
    q: &DMat<f64>,
    r: &DMat<f64>,
) -> Result<LqrTriple, LinalgError> {
    let sol = dlqr_solve(a, b, q, r)?;
    let e = closed_loop_eigs(a, b, &sol.k)?;
    Ok((sol.k, sol.x, e))
}

/// `[L,P] = kalman(A,C,Qn,Rn)`: steady-state Kalman predictor-form gain `L`
/// and error covariance `P`.
pub fn kalman(
    a: &DMat<f64>,
    c: &DMat<f64>,
    qn: &DMat<f64>,
    rn: &DMat<f64>,
) -> Result<(DMat<f64>, DMat<f64>), LinalgError> {
    let (p, l) = kalman_gain(a, c, qn, rn)?;
    Ok((l, p))
}

fn closed_loop_eigs(a: &DMat<f64>, b: &DMat<f64>, k: &DMat<f64>) -> Result<Vec<C64>, LinalgError> {
    poles(&(a - &(b * k)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf2ss_of_first_order_system() {
        // 1/(s+2): A = -2, B = 1, C = 1, D = 0.
        let (a, b, c, d) = tf2ss(&[1.0], &[1.0, 2.0]).unwrap();
        assert!((a[(0, 0)] + 2.0).abs() < 1e-12);
        assert!((b[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((c[(0, 0)] - 1.0).abs() < 1e-12);
        assert!(d[(0, 0)].abs() < 1e-12);
    }

    #[test]
    fn lqr_returns_matlab_triple() {
        let a = DMat::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]);
        let b = DMat::from_rows(&[&[0.0], &[1.0]]);
        let q = DMat::identity(2);
        let r = DMat::from_rows(&[&[1.0]]);
        let (k, s, e) = lqr(&a, &b, &q, &r).unwrap();
        assert!((k[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((k[(0, 1)] - 3.0f64.sqrt()).abs() < 1e-6);
        assert!((s[(0, 0)] - 3.0f64.sqrt()).abs() < 1e-6);
        assert_eq!(e.len(), 2);
        assert!(e.iter().all(|p| p.re < 0.0));
    }

    #[test]
    fn c2d_matches_zoh() {
        let a = DMat::from_rows(&[&[-1.0]]);
        let b = DMat::from_rows(&[&[1.0]]);
        let (ad, bd) = c2d(&a, &b, 0.5).unwrap();
        assert!((ad[(0, 0)] - (-0.5f64).exp()).abs() < 1e-12);
        assert!((bd[(0, 0)] - (1.0 - (-0.5f64).exp())).abs() < 1e-12);
    }
}
